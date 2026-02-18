//! A solver for the travelling salesman problem.
//!
//! Solves euclidean TPS problems using the integer linear programming approach.
//! See comments in the solve() function for the detailed description of the algorithm.

use microlp::problems_solvers::tsp::{solve_tsp, TspProblem};
use std::io;

const USAGE: &str = "\
USAGE:
    tsp --help
    tsp [--svg-output] INPUT_FILE

INPUT_FILE is a problem description in TSPLIB format. You can download some
problems from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/.
Use - for stdin.

By default, prints a single line containing 1-based node indices in the
optimal tour order to stdout. If --svg-output option is enabled, prints an
SVG document containing the optimal tour.

Set RUST_LOG environment variable (e.g. to info) to enable logging to stderr.
";

fn main() {
    env_logger::init();

    let args = std::env::args().collect::<Vec<_>>();
    if args.len() <= 1 {
        eprint!("{}", USAGE);
        std::process::exit(1);
    }

    if args[1] == "--help" {
        eprintln!("Finds the optimal solution for a traveling salesman problem.\n");
        eprint!("{}", USAGE);
        return;
    }

    let (enable_svg_output, filename) = if args.len() == 2 {
        (false, &args[1])
    } else if args.len() == 3 && args[1] == "--svg-output" {
        (true, &args[2])
    } else {
        eprintln!("Failed to parse arguments.\n");
        eprint!("{}", USAGE);
        std::process::exit(1);
    };

    let problem = if filename == "-" {
        TspProblem::parse(std::io::stdin().lock()).unwrap()
    } else {
        let file = std::fs::File::open(filename).unwrap();
        let input = io::BufReader::new(file);
        TspProblem::parse(input).unwrap()
    };

    let tour = solve_tsp(&problem);
    if enable_svg_output {
        print!("{}", tour.to_svg(&problem));
    } else {
        println!("{}", tour.to_string());
    }
}
