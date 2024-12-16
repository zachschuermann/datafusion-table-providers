use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::datasource::{provider_as_source, TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{
    project_schema, DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion::prelude::*;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::Table;
use itertools::Itertools;
use tokio::time::timeout;

// ref: https://datafusion.apache.org/library-user-guide/custom-table-providers.html
// ref: https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/custom_datasource.rs

/// This example demonstrates executing a simple query against a custom datasource
#[tokio::main]
async fn main() -> Result<()> {
    let table = DeltaDataSource::new("examples/delta_example/");
    // create local execution context
    let ctx = SessionContext::new();

    // create logical plan composed of a single TableScan
    let logical_plan = LogicalPlanBuilder::scan_with_filters(
        "delta_example",
        provider_as_source(Arc::new(table)),
        None,
        vec![],
    )?
    .build()?;

    let dataframe = DataFrame::new(ctx.state(), logical_plan).select_columns(&["value"])?;

    // if let Some(f) = filter {
    //     dataframe = dataframe.filter(f)?;
    // }

    timeout(Duration::from_secs(10), async move {
        let result = dataframe.collect().await.unwrap();
        let record_batch = result.first().unwrap();

        // assert_eq!(3, record_batch.column(1).len());
        dbg!(record_batch.columns());
    })
    .await
    .unwrap();

    Ok(())
}

#[derive(Clone)]
pub struct DeltaDataSource {
    table: Table,
    engine: Arc<DefaultEngine<TokioBackgroundExecutor>>,
}

impl Debug for DeltaDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("delta_table")
    }
}

impl DeltaDataSource {
    pub fn new(table_path: impl ToString) -> Self {
        let table = Table::try_from_uri(table_path.to_string()).unwrap();
        let engine = Arc::new(
            DefaultEngine::try_new(
                table.location(),
                HashMap::<&str, &str>::new(),
                Arc::new(TokioBackgroundExecutor::new()),
            )
            .unwrap(),
        );

        Self { table, engine }
    }

    pub(crate) async fn create_physical_plan(
        &self,
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(DeltaExec::new(projections, schema, self.clone())))
    }
}

#[async_trait]
impl TableProvider for DeltaDataSource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        SchemaRef::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt32,
            false,
        )]))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        // filters and limit can be used here to inject some push-down operations if needed
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        return self.create_physical_plan(projection, self.schema()).await;
    }
}

#[derive(Debug, Clone)]
struct DeltaExec {
    delta_data_source: DeltaDataSource,
    projected_schema: SchemaRef,
    cache: PlanProperties,
}

impl DeltaExec {
    fn new(
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
        delta_data_source: DeltaDataSource,
    ) -> Self {
        let projected_schema = project_schema(&schema, projections).unwrap();
        let cache = Self::compute_properties(projected_schema.clone());
        Self {
            delta_data_source,
            projected_schema,
            cache,
        }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);
        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        )
    }
}

impl DisplayAs for DeltaExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "DeltaExec")
    }
}

impl ExecutionPlan for DeltaExec {
    fn name(&self) -> &'static str {
        "DeltaExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // read latest snapshot
        let snapshot = self
            .delta_data_source
            .table
            .snapshot(self.delta_data_source.engine.as_ref(), None)
            .unwrap();
        dbg!(snapshot.version());

        // let schema = snapshot.schema();

        // TODO support predicate
        let scan = snapshot
            .into_scan_builder()
            .with_schema(Arc::new(self.projected_schema.as_ref().try_into().unwrap()))
            .build()
            .unwrap();

        let batches = scan
            .execute(self.delta_data_source.engine.as_ref())
            .unwrap()
            .map(|scan_result| -> delta_kernel::DeltaResult<_> {
                let scan_result = scan_result?;
                let mask = scan_result.full_mask();
                let data = scan_result.raw_data?;
                let record_batch: RecordBatch = data
                    .into_any()
                    .downcast::<ArrowEngineData>()
                    .map_err(|_| {
                        delta_kernel::Error::EngineDataType("ArrowEngineData".to_string())
                    })?
                    .into();
                if let Some(mask) = mask {
                    // FIXME: should use `FilterExec`?
                    use datafusion::common::arrow::compute::filter_record_batch;
                    Ok(filter_record_batch(&record_batch, &mask.into())?)
                } else {
                    Ok(record_batch)
                }
            });

        // seems like we should be able to avoid collect and stream?
        Ok(Box::pin(MemoryStream::try_new(
            batches.try_collect().unwrap(),
            self.schema(),
            None,
        )?))
    }
}

// /// This example demonstrates how to register a custom Delta table provider and register a local
// /// table to perform a scan. TODO normalize with other examples. (consider connection pool and
// /// table factory)
// // #[tokio::main]
// async fn main() {
//     // Create DataFusion session context
//     let ctx = SessionContext::new();
//
//     let delta_table_provider = DeltaDataSource::new();
//     ctx.register_table("delta_table", Arc::new(delta_table_provider))
//         .expect("failed to register table");
//
//     let df = ctx
//         .sql("SELECT * FROM delta_table")
//         .await
//         .expect("select failed");
//
//     df.show().await.expect("show failed");
// }
