//! The column ordering of the LU factorization: [`order_simple`], which takes
//! the columns by their number of entries, the sparsest first, out of the
//! bucket queue in [`cols_queue`]. `lu_factorize` uses it for every
//! factorization, and `lu_factorize_ranked` [`order_ranked`], which sorts it
//! by a rank of the columns.

mod cols_queue;

use cols_queue::ColsQueue;

use crate::sparse::Perm;

/// Simplest preordering: order columns based on their size. An empty column
/// ranks with the columns of one entry; the factorization finds it dependent.
pub fn order_simple<'a>(size: usize, get_col: impl Fn(usize) -> &'a [usize]) -> Perm {
    let mut cols_queue = ColsQueue::new(size);
    for c in 0..size {
        cols_queue.add(c, get_col(c).len().saturating_sub(1));
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

    Perm { orig2new, new2orig }
}

/// [`order_simple`], stably sorted by `rank`, the lowest first.
pub fn order_ranked<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
    rank: impl Fn(usize) -> u8,
) -> Perm {
    let Perm { mut new2orig, .. } = order_simple(size, get_col);
    new2orig.sort_by_key(|&c| rank(c));
    let mut orig2new = vec![0; size];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }
    Perm { orig2new, new2orig }
}

#[cfg(test)]
mod tests;
