//TODO add more tests
// https://github.com/rhgrant10/tsplib95/tree/master/archives/problems
#[cfg(test)]
mod tests_tsp {
    use crate::problems_solvers::tsp::{solve_tsp, TspProblem};
    use std::fs;
    use std::io::BufReader;
    use std::path::PathBuf;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn inputs_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("tests")
            .join("tsp")
            .join("inputs")
    }

    fn discover_tsp_cases() -> Vec<(usize, PathBuf, PathBuf)> {
        let dir = inputs_dir();
        let mut cases = Vec::new();

        for entry in fs::read_dir(&dir).expect("failed to read tsp inputs directory") {
            let entry = entry.expect("failed to read directory entry");
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(num_str) = name
                    .strip_prefix("tsp")
                    .and_then(|s| s.strip_suffix(".tsp"))
                {
                    if let Ok(num) = num_str.parse::<usize>() {
                        let out_path = dir.join(format!("tsp{}-out.txt", num));
                        if out_path.exists() {
                            cases.push((num, path.clone(), out_path));
                        } else {
                            panic!(
                                "Found input file {} but no matching output file {}",
                                path.display(),
                                out_path.display()
                            );
                        }
                    }
                }
            }
        }

        cases.sort_by_key(|(num, _, _)| *num);
        assert!(
            !cases.is_empty(),
            "No tsp test cases found in {}",
            dir.display()
        );
        cases
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "test is too slow in debug mode")]
    fn solve_all_tsp_cases() {
        init();

        let cases = discover_tsp_cases();

        for (num, input_path, output_path) in &cases {
            println!("Running TSP test case {}: {}", num, input_path.display());

            let input_file = fs::File::open(input_path)
                .expect(&format!("failed to open {}", input_path.display()));
            let reader = BufReader::new(input_file);
            let problem = TspProblem::parse(reader)
                .expect(&format!("failed to parse {}", input_path.display()));

            let tour = solve_tsp(&problem);
            let actual = tour.to_string();

            let expected = fs::read_to_string(output_path)
                .expect(&format!("failed to read {}", output_path.display()))
                .trim()
                .to_string();

            assert_eq!(
                actual, expected,
                "TSP test case {} failed.\nExpected: {}\nActual:   {}",
                num, expected, actual
            );

            println!("TSP test case {} passed.", num);
        }
    }
}
