/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * // SPDX-License-Identifier: BSD
 */

use mountpoint_s3_client::types::HeadObjectResult;
use pyo3::types::PyTuple;
use pyo3::{pyclass, pymethods};
use pyo3::{Bound, ToPyObject};
use pyo3::{IntoPy, PyResult};

use crate::python_structs::py_restore_status::PyRestoreStatus;
use crate::PyRef;

#[pyclass(
    name = "HeadObjectResult",
    module = "s3torchconnectorclient._mountpoint_s3_client",
    frozen
)]
#[derive(Debug, Clone)]
pub struct PyHeadObjectResult {
    #[pyo3(get)]
    etag: String,
    #[pyo3(get)]
    size: u64,
    #[pyo3(get)]
    last_modified: i64,
    #[pyo3(get)]
    storage_class: Option<String>,
    #[pyo3(get)]
    restore_status: Option<PyRestoreStatus>,
}

impl PyHeadObjectResult {
    pub(crate) fn from_head_object_result(head_object_result: HeadObjectResult) -> Self {
        PyHeadObjectResult::new(
            head_object_result.etag.into_inner(),
            head_object_result.size,
            head_object_result.last_modified.unix_timestamp(),
            head_object_result.storage_class,
            head_object_result
                .restore_status
                .map(PyRestoreStatus::from_restore_status),
        )
    }
}

#[pymethods]
impl PyHeadObjectResult {
    #[new]
    #[pyo3(signature = (etag, size, last_modified, storage_class=None, restore_status=None))]
    pub fn new(
        etag: String,
        size: u64,
        last_modified: i64,
        storage_class: Option<String>,
        restore_status: Option<PyRestoreStatus>,
    ) -> Self {
        Self {
            etag,
            size,
            last_modified,
            storage_class,
            restore_status,
        }
    }

    #[allow(clippy::useless_conversion)]
    pub fn __getnewargs__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyTuple>> {
        let py = slf.py();
        let state = [
            slf.etag.to_object(py),
            slf.size.to_object(py),
            slf.last_modified.to_object(py),
            slf.storage_class.to_object(py),
            slf.restore_status.clone().into_py(py),
        ];
        Ok(PyTuple::new_bound(py, state))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}
