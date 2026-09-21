//! Column orderings for the LU factorization, plus the structural analyses
//! built on the same sparsity pattern.
//!
//! * [`order_simple`] — order by column size; cheap, used as the fallback.
//! * [`colamd`] — approximate minimum degree, kept for when the solver
//!   switches orderings.
//! * [`cols_queue`] — the bucket queue both orderings pick columns from.
//! * [`matching`] — diagonal matching and lower block triangular form.

mod colamd;
mod cols_queue;
mod matching;

use cols_queue::ColsQueue;

use crate::sparse::{Error, Perm};

/// Simplest preordering: order columns based on their size
pub fn order_simple<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> Result<Perm, Error> {
    let mut cols_queue = ColsQueue::new(size);
    for c in 0..size {
        let col = get_col(c);
        if col.is_empty() {
            return Err(Error::SingularMatrix);
        }
        cols_queue.add(c, col.len() - 1);
    }

    let mut new2orig = Vec::with_capacity(size);

    //TODO should this be refactored?
    while new2orig.len() < size {
        let min = cols_queue.pop_min();
        //guaranteed to exist
        new2orig.push(min.unwrap());
    }

    let mut orig2new = vec![0; size];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    Ok(Perm { orig2new, new2orig })
}

#[cfg(test)]
mod tests;
