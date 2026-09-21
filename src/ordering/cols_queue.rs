//! [`ColsQueue`]: a bucket queue over column scores, so "the cheapest
//! remaining column" is an O(1) lookup for both orderings.

#[derive(Clone, Debug)]
pub(super) struct Slice {
    pub(super) begin: usize,
    pub(super) end: usize,
}

impl Slice {
    pub(super) fn elems<'a>(&self, storage: &'a [usize]) -> &'a [usize] {
        &storage[self.begin..self.end]
    }
}

#[derive(Debug)]
pub(super) struct ColsQueue {
    score2head: Vec<Option<usize>>,
    prev: Vec<usize>,
    next: Vec<usize>,
    min_score: usize,
    pub(super) len: usize,
}

impl ColsQueue {
    pub(super) fn new(num_cols: usize) -> ColsQueue {
        ColsQueue {
            score2head: vec![None; num_cols],
            prev: vec![0; num_cols],
            next: vec![0; num_cols],
            min_score: num_cols,
            len: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn pop_min(&mut self) -> Option<usize> {
        let col = loop {
            if self.min_score >= self.score2head.len() {
                //TODO this branch is not hit
                return None;
            }
            if let Some(col) = self.score2head[self.min_score] {
                break col;
            }
            self.min_score += 1;
        };

        self.remove(col, self.min_score);
        Some(col)
    }

    pub(super) fn add(&mut self, col: usize, score: usize) {
        self.min_score = std::cmp::min(self.min_score, score);
        self.len += 1;

        if let Some(head) = self.score2head[score] {
            self.prev[col] = self.prev[head];
            self.next[col] = head;
            self.next[self.prev[head]] = col;
            self.prev[head] = col;
        } else {
            self.prev[col] = col;
            self.next[col] = col;
            self.score2head[score] = Some(col);
        }
    }

    pub(super) fn remove(&mut self, col: usize, score: usize) {
        self.len -= 1;
        if self.next[col] == col {
            self.score2head[score] = None;
        } else {
            self.next[self.prev[col]] = self.next[col];
            self.prev[self.next[col]] = self.prev[col];

            //will panic if score is not valid
            if self.score2head[score].unwrap() == col {
                self.score2head[score] = Some(self.next[col]);
            }
        }
    }
}
