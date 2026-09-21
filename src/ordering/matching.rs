//! Structural matching and block decomposition: pairing each column with a row
//! it has a nonzero in, and splitting the result into the strongly connected
//! blocks of a lower block triangular form.

const SENTINEL: usize = 0usize.wrapping_sub(1);

#[allow(dead_code)]
pub fn find_diag_matching<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> Option<Vec<usize>> {
    let mut col2visited_on_iter = vec![SENTINEL; size];
    let mut row2matched_col = vec![SENTINEL; size];
    // for each col a pointer to the position in its adjacency lists where we last looked for neighbors.
    let mut cheap = vec![0; size];

    struct Step {
        col: usize,
        cur_i: usize,
    }

    let mut dfs_stack = vec![];
    for start_c in 0..size {
        let mut found = false; // whether the dfs iteration found the match

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });

        'dfs_loop: while !dfs_stack.is_empty() {
            //guaranteed to exist
            let cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            let col_rows = get_col(c);

            if col2visited_on_iter[c] != start_c {
                col2visited_on_iter[c] = start_c;

                let cur_cheap = &mut cheap[c];
                while *cur_cheap < col_rows.len() {
                    let r = col_rows[*cur_cheap];
                    if row2matched_col[r] == SENTINEL {
                        row2matched_col[r] = c;
                        found = true;
                        dfs_stack.pop();
                        continue 'dfs_loop;
                    }
                    *cur_cheap += 1;
                }
            } else {
                if found {
                    let r = col_rows[cur_step.cur_i];
                    row2matched_col[r] = c;
                    dfs_stack.pop();
                    continue 'dfs_loop;
                }

                cur_step.cur_i += 1;
            }

            while cur_step.cur_i < col_rows.len() {
                let r = col_rows[cur_step.cur_i];
                if col2visited_on_iter[row2matched_col[r]] != start_c {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i == col_rows.len() {
                dfs_stack.pop();
            } else {
                let col = row2matched_col[col_rows[cur_step.cur_i]];
                dfs_stack.push(Step { col, cur_i: 0 });
            }
        }

        if !found {
            return None;
        }
    }

    Some(row2matched_col)
}

/// Lower block triangular form of a matrix.
#[derive(Clone, Debug)]
pub struct BlockDiagForm {
    #[allow(unused)]
    /// Row permutation: for each original row its new row number so that diag is nonzero.
    pub row2col: Vec<usize>,
    #[allow(unused)]
    /// For each block its set of columns (the order of blocks is lower block triangular)
    pub block_cols: Vec<Vec<usize>>,
}

#[allow(dead_code)]
pub fn find_block_diag_form<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> BlockDiagForm {
    //TODO check unwrap
    let row2col = find_diag_matching(size, &get_col).unwrap();

    struct Step {
        col: usize,
        cur_i: usize,
    }

    let mut dfs_stack = vec![];
    let mut visited = vec![];
    let mut is_visited = vec![false; size];
    for start_c in 0..size {
        if is_visited[start_c] {
            continue;
        }

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });
        while !dfs_stack.is_empty() {
            //guaranteed to exist
            let cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            if !is_visited[c] {
                is_visited[c] = true;
            } else {
                cur_step.cur_i += 1;
            }

            let col_rows = get_col(c);
            while cur_step.cur_i < col_rows.len() {
                let next_c = row2col[col_rows[cur_step.cur_i]];
                if !is_visited[next_c] {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i < col_rows.len() {
                let col = row2col[col_rows[cur_step.cur_i]];
                dfs_stack.push(Step { col, cur_i: 0 });
            } else {
                visited.push(c);
                dfs_stack.pop();
            }
        }
    }

    // Prepare transposed graph
    // TODO: more efficient transpose without allocating each row.
    let mut rows = vec![vec![]; size];
    for c in 0..size {
        for &r in get_col(c) {
            rows[row2col[r]].push(c);
        }
    }

    is_visited.clear();
    is_visited.resize(size, false);

    let mut block_cols = vec![];

    // DFS on the transposed graph
    for &start_c in visited.iter().rev() {
        if is_visited[start_c] {
            continue;
        }

        block_cols.push(vec![]);

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });
        while !dfs_stack.is_empty() {
            //guaranteed to exist
            let cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            if !is_visited[c] {
                is_visited[c] = true;
                //guaranteed to exist, has at least one element
                block_cols.last_mut().unwrap().push(c);
            } else {
                cur_step.cur_i += 1;
            }

            let next = &rows[c];
            while cur_step.cur_i < next.len() {
                if !is_visited[next[cur_step.cur_i]] {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i < next.len() {
                let col = next[cur_step.cur_i];
                dfs_stack.push(Step { col, cur_i: 0 });
            } else {
                dfs_stack.pop();
            }
        }
    }

    BlockDiagForm {
        row2col,
        block_cols,
    }
}
