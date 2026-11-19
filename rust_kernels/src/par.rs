//! The worker pool every kernel shares, and the few helpers built on it.

use rayon::prelude::*;
use std::sync::{Arc, RwLock};

static POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

fn default_threads() -> usize {
    if let Ok(v) = std::env::var("GRAVITY_OCR_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn build(n: usize) -> Arc<rayon::ThreadPool> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .thread_name(|i| format!("gk-{i}"))
        .build()
        .expect("gravity_kernels: cannot start the worker pool");
    Arc::new(pool)
}

pub fn pool() -> Arc<rayon::ThreadPool> {
    if let Some(p) = POOL.read().unwrap().as_ref() {
        return p.clone();
    }
    let mut slot = POOL.write().unwrap();
    if slot.is_none() {
        *slot = Some(build(default_threads()));
    }
    slot.as_ref().unwrap().clone()
}

/// Replace the pool; `0` goes back to the default.  Returns the new size.
pub fn set_threads(n: usize) -> usize {
    let pool = build(if n == 0 { default_threads() } else { n });
    let size = pool.current_num_threads();
    *POOL.write().unwrap() = Some(pool);
    size
}

pub fn threads() -> usize {
    pool().current_num_threads()
}

/// `f(0) .. f(n - 1)`, on the pool when that can help.
pub fn run<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
    if n == 0 {
        return;
    }
    let pool = pool();
    if n == 1 || pool.current_num_threads() == 1 {
        for i in 0..n {
            f(i);
        }
        return;
    }
    pool.install(|| (0..n).into_par_iter().for_each(|i| f(i)));
}

/// `f(first, last)` over `0..n` cut into pieces of `step`, in parallel.
pub fn chunks<F: Fn(usize, usize) + Sync + Send>(n: usize, step: usize, f: F) {
    let step = step.max(1);
    run(n.div_ceil(step), |i| f(i * step, ((i + 1) * step).min(n)));
}

/// A raw pointer the tasks of one call may share; they write disjoint places.
pub struct Ptr<T>(pub *mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Ptr<T> {}
impl<T> Ptr<T> {
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

pub struct Cptr<T>(pub *const T);
unsafe impl<T> Send for Cptr<T> {}
unsafe impl<T> Sync for Cptr<T> {}
impl<T> Clone for Cptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Cptr<T> {}
impl<T> Cptr<T> {
    #[inline]
    pub fn get(self) -> *const T {
        self.0
    }
}

/// A 64-byte aligned, zeroed f32 buffer.
pub struct Buf {
    ptr: *mut f32,
    len: usize,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn new(len: usize) -> Buf {
        if len == 0 {
            return Buf { ptr: std::ptr::NonNull::<f32>::dangling().as_ptr(), len: 0 };
        }
        let layout = std::alloc::Layout::from_size_align(len * 4, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Buf { ptr, len }
    }

    #[inline]
    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.len != 0 {
            let layout = std::alloc::Layout::from_size_align(self.len * 4, 64).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::UnsafeCell<Buf> = std::cell::UnsafeCell::new(Buf::new(0));
}

/// A per-thread scratch buffer of at least `len` floats (contents undefined).
///
/// The closure must not call back into the pool: the buffer belongs to the
/// thread, not to the task.
pub fn with_scratch<R>(len: usize, f: impl FnOnce(*mut f32) -> R) -> R {
    SCRATCH.with(|cell| {
        let buf = unsafe { &mut *cell.get() };
        if buf.len() < len {
            *buf = Buf::new(len);
        }
        f(buf.ptr())
    })
}
