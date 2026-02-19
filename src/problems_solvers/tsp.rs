//disable clippy
//

use crate::{ComparisonOp, LinearExpr, OptimizationDirection, Variable};
use std::io;

/// Solve the Travelling Salesman Problem using integer linear programming with iterative subtour elimination.
#[allow(clippy::all)]
pub fn solve_tsp(problem: &TspProblem) -> Tour {
    info!("starting, problem name: {}", problem.name);

    let num_points = problem.points.len();

    // First, we construct an integer linear programming model for the TSP problem.
    let mut lp_problem = crate::Problem::new(OptimizationDirection::Minimize);

    // Variables in our model correspond to edges between nodes (cities). If the tour includes
    // the edge between nodes i and j, then the edge_vars[i][j] variable will be equal to 1
    // in the solution. We use binary variables so that the solver's built-in branch & bound
    // handles integrality for us. Each edge will contribute the distance between its endpoints
    // to the objective function which we want to minimize.
    let mut edge_vars = vec![vec![]; num_points];
    for i in 0..num_points {
        for j in 0..num_points {
            let var = if j < i {
                edge_vars[j][i]
            } else {
                lp_problem.add_binary_var(problem.dist(i, j))
            };
            edge_vars[i].push(var);
        }
    }

    // Next, we add constraints that will ensure that each node is part of a tour.
    // To do this we specify that for each node exactly two edges incident on that node
    // are present in the solution. Or, equivalently: for every node the sum of all edge
    // variables incident on that node is equal to 2.0.
    for i in 0..num_points {
        let mut edges_sum = LinearExpr::empty();
        for j in 0..num_points {
            if i != j {
                edges_sum.add(edge_vars[i][j], 1.0);
            }
        }
        lp_problem.add_constraint(edges_sum, ComparisonOp::Eq, 2.0);
    }

    // We solve the integer problem iteratively using a cutting-plane approach for subtour
    // elimination. Nothing in the model so far prohibits *subtours* - closed tours that pass
    // through only a subset of all nodes. Unfortunately, to prohibit all subtours we would
    // need exponentially many constraints. Instead, we solve the problem, check for subtours
    // in the integer solution, add violated subtour elimination constraints, and re-solve.
    // The solver's built-in branch & bound handles integrality at each iteration.

    loop {
        let solution = lp_problem.solve().unwrap();
        info!(
            "solved integer problem, obj. value: {:.2}",
            solution.objective(),
        );

        // Find subtours in the integer solution. If there is exactly one tour covering
        // all nodes, we have found the optimal solution.
        let subtours = find_subtours(&solution, &edge_vars);
        if subtours.len() == 1 {
            info!("found optimal solution, cost: {:.2}", solution.objective());
            return tour_from_lp_solution(&solution, &edge_vars);
        }

        info!(
            "solution has {} subtours, adding elimination constraints",
            subtours.len()
        );

        // For each subtour, add a constraint requiring that at least 2 edges cross
        // the boundary of the subtour's node set.
        for subtour in &subtours {
            let mut in_subtour = vec![false; num_points];
            for &node in subtour {
                in_subtour[node] = true;
            }

            let mut cut_edges_sum = LinearExpr::empty();
            for i in 0..num_points {
                for j in 0..i {
                    if in_subtour[i] != in_subtour[j] {
                        cut_edges_sum.add(edge_vars[i][j], 1.0);
                    }
                }
            }
            lp_problem.add_constraint(cut_edges_sum, ComparisonOp::Ge, 2.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn sqr_dist(self, other: Self) -> f64 {
        (self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (self.y - other.y)
    }

    fn dist(self, other: Self) -> f64 {
        f64::sqrt(self.sqr_dist(other))
    }
}

/// A TSP problem instance with named 2D Euclidean points.
#[derive(Debug)]
pub struct TspProblem {
    /// The name of the problem instance.
    pub name: String,
    points: Vec<Point>,
}

fn read_line<R: io::BufRead>(mut input: R) -> io::Result<Vec<String>> {
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(line.split_whitespace().map(|tok| tok.to_owned()).collect())
}

fn parse_num<T: std::str::FromStr>(input: &str, line_num: usize) -> io::Result<T> {
    input.parse::<T>().or(Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("line {}: couldn't parse number", line_num),
    )))
}

impl TspProblem {
    /// Returns the Euclidean distance between two nodes.
    pub fn dist(&self, n1: usize, n2: usize) -> f64 {
        self.points[n1].dist(self.points[n2])
    }

    /// Parse a problem in the TSPLIB format.
    ///
    /// Format description: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    pub fn parse<R: io::BufRead>(mut input: R) -> io::Result<TspProblem> {
        let mut name = String::new();
        let mut dimension = None;
        let mut line_num = 0;
        loop {
            let line = read_line(&mut input)?;
            if line.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "premature end of header".to_string(),
                ));
            }

            let mut keyword = line[0].clone();
            if keyword.ends_with(":") {
                keyword.pop();
            }

            if keyword == "NAME" {
                name = line.last().unwrap().clone();
            } else if keyword == "TYPE" {
                if line.last().unwrap() != "TSP" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "only problems with TYPE: TSP supported".to_string(),
                    ));
                }
            } else if keyword == "EDGE_WEIGHT_TYPE" {
                if line.last().unwrap() != "EUC_2D" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "only problems with EDGE_WEIGHT_TYPE: EUC_2D supported".to_string(),
                    ));
                }
            } else if keyword == "DIMENSION" {
                let dim: usize = parse_num(line.last().as_ref().unwrap(), line_num)?;
                dimension = Some(dim);
            } else if keyword == "NODE_COORD_SECTION" {
                break;
            }

            line_num += 1;
        }

        let num_points = dimension.ok_or(io::Error::new(
            io::ErrorKind::InvalidData,
            "no DIMENSION specified".to_string(),
        ))?;
        if num_points > 100_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("problem dimension: {} is suspiciously large", num_points),
            ));
        }

        let mut point_opts = vec![None; num_points];
        for _ in 0..num_points {
            let line = read_line(&mut input)?;
            let node_num: usize = parse_num(&line[0], line_num)?;
            let x: f64 = parse_num(&line[1], line_num)?;
            let y: f64 = parse_num(&line[2], line_num)?;
            if node_num == 0 || node_num > num_points {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: bad node number: {}", line_num, node_num),
                ));
            }
            if point_opts[node_num - 1].is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: node {} specified twice", line_num, node_num),
                ));
            }
            point_opts[node_num - 1] = Some(Point { x, y });

            line_num += 1;
        }

        let line = read_line(input)?;
        if line.len() > 1 || (line.len() == 1 && line[0] != "EOF") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: expected EOF", line_num),
            ));
        }

        let mut points = vec![];
        for (i, po) in point_opts.into_iter().enumerate() {
            if let Some(p) = po {
                points.push(p);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("node {} is not specified", i),
                ));
            }
        }

        Ok(TspProblem { name, points })
    }
}

/// A solution to the TSP problem: a sequence of 0-based node indices in the tour order.
/// Each node must be present in the tour exactly once.
#[derive(Debug)]
pub struct Tour(Vec<usize>);

impl Tour {
    /// Returns a space-separated string of 1-based node indices in tour order.
    #[allow(clippy::all)]
    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|n| (n + 1).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Renders the tour as an SVG document.
    pub fn to_svg(&self, problem: &TspProblem) -> String {
        let cmp_f64 = |x: &f64, y: &f64| x.partial_cmp(y).unwrap();
        let min_x = problem.points.iter().map(|p| p.x).min_by(cmp_f64).unwrap();
        let max_x = problem.points.iter().map(|p| p.x).max_by(cmp_f64).unwrap();
        let min_y = problem.points.iter().map(|p| p.y).min_by(cmp_f64).unwrap();
        let max_y = problem.points.iter().map(|p| p.y).max_by(cmp_f64).unwrap();

        let width = 600;
        let margin = 50;
        let scale = ((width - 2 * margin) as f64) / (max_x - min_x);
        let height = f64::round((max_y - min_y) * scale) as usize + 2 * margin;

        let mut svg = String::new();
        svg += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
        svg += "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n";
        svg += "  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
        svg += &format!(
            "<svg width=\"{}px\" height=\"{}px\" version=\"1.1\"",
            width, height
        );
        svg += "     xmlns=\"http://www.w3.org/2000/svg\">\n";

        use std::fmt::Write;
        svg += "    <path fill=\"none\" stroke=\"black\" stroke-width=\"4px\" d=\"\n";
        for &i in &self.0 {
            let p = problem.points[i];
            let px = f64::round((p.x - min_x) * scale) as usize + margin;
            let py = f64::round((p.y - min_y) * scale) as usize + margin;
            if i == 0 {
                writeln!(&mut svg, "        M {} {}", px, py).unwrap();
            } else {
                writeln!(&mut svg, "        L {} {}", px, py).unwrap();
            }
        }
        svg += "        Z\n";
        svg += "    \"/>\n";

        svg += "</svg>\n";
        svg
    }
}

/// Find all subtours in an integer solution. Each subtour is a vector of node indices
/// forming a closed cycle. If the solution is a valid TSP tour, there will be exactly
/// one subtour containing all nodes.
fn find_subtours(solution: &crate::Solution, edge_vars: &[Vec<Variable>]) -> Vec<Vec<usize>> {
    let num_points = edge_vars.len();
    let mut visited = vec![false; num_points];
    let mut subtours = vec![];

    for start in 0..num_points {
        if visited[start] {
            continue;
        }

        let mut tour = vec![];
        let mut cur = start;
        loop {
            visited[cur] = true;
            tour.push(cur);
            let mut found_next = false;
            for neighbor in 0..num_points {
                if !visited[neighbor] && solution[edge_vars[cur][neighbor]].round() == 1.0 {
                    cur = neighbor;
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }

        subtours.push(tour);
    }

    subtours
}

/// Convert a solution to the LP problem to the corresponding tour (a sequence of nodes).
/// Precondition: the solution must be integral and contain a unique tour.
fn tour_from_lp_solution(lp_solution: &crate::Solution, edge_vars: &[Vec<Variable>]) -> Tour {
    let num_points = edge_vars.len();
    let mut tour = vec![];
    let mut is_visited = vec![false; num_points];
    let mut cur_point = 0;
    for _ in 0..num_points {
        assert!(!is_visited[cur_point]);
        is_visited[cur_point] = true;
        tour.push(cur_point);
        for neighbor in 0..num_points {
            if !is_visited[neighbor] && lp_solution[edge_vars[cur_point][neighbor]].round() == 1.0 {
                cur_point = neighbor;
                break;
            }
        }
    }
    assert_eq!(tour.len(), num_points);
    Tour(tour)
}
