use std::collections::{HashMap, HashSet};

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Port of the "process nodes" loop from populate_communities().
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, community_composition, community_sizes, group_exposure, ideal, target_counts=None, total_nodes=0))]
fn process_nodes<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    mut community_composition: PyReadwriteArray2<'py, f64>,
    mut community_sizes: PyReadwriteArray1<'py, i32>,
    mut group_exposure: PyReadwriteArray2<'py, f64>,
    ideal: PyReadonlyArray2<'py, f64>,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let all_nodes = all_nodes.as_array();
    let node_groups = node_groups.as_array();
    let ideal = ideal.as_array();

    let num_communities = community_composition.as_array().shape()[0];
    let n_groups = community_composition.as_array().shape()[1];

    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());

    let mut rng = thread_rng();
    let mut assignments: Vec<i64> = Vec::with_capacity(total_nodes);

    let mut distances = vec![0.0f64; num_communities];
    let mut hyp_row = vec![0.0f64; n_groups];

    for node_idx in 0..total_nodes {
        let group = node_groups[node_idx] as usize;

        let comp = community_composition.as_array();
        let ge = group_exposure.as_array();

        for c in 0..num_communities {
            let mut hyp_total: f64 = 0.0;
            for g in 0..n_groups {
                let val = ge[[group, g]] + comp[[c, g]];
                hyp_row[g] = val;
                hyp_total += val;
            }
            if hyp_total < 1e-10 {
                hyp_total = 1e-10;
            }
            let mut dist_sq: f64 = 0.0;
            for g in 0..n_groups {
                let diff = (hyp_row[g] / hyp_total) - ideal[[group, g]];
                dist_sq += diff * diff;
            }
            distances[c] = dist_sq.sqrt();
        }

        if let Some(ref tc) = tc {
            let sizes = community_sizes.as_array();
            for c in 0..num_communities {
                if sizes[c] >= tc[c] {
                    distances[c] = f64::INFINITY;
                }
            }
        }

        let temperature: f64 = 1.0 - (node_idx as f64 / total_nodes as f64);
        let best_community: usize;

        if temperature > 0.05 {
            let valid: Vec<usize> = (0..num_communities)
                .filter(|&c| distances[c].is_finite())
                .collect();

            if valid.len() > 1 {
                let max_neg_d = valid
                    .iter()
                    .map(|&c| -distances[c] / (temperature + 1e-10))
                    .fold(f64::NEG_INFINITY, f64::max);

                let weights: Vec<f64> = valid
                    .iter()
                    .map(|&c| ((-distances[c] / (temperature + 1e-10)) - max_neg_d).exp())
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                best_community = valid[dist.sample(&mut rng)];
            } else if valid.len() == 1 {
                best_community = valid[0];
            } else {
                best_community = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
            }
        } else {
            best_community = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
        }

        assignments.push(best_community as i64);

        {
            let comp = community_composition.as_array();
            let mut ge = group_exposure.as_array_mut();
            for g in 0..n_groups {
                ge[[group, g]] += comp[[best_community, g]];
            }
            for g in 0..n_groups {
                if comp[[best_community, g]] > 0.0 {
                    ge[[g, group]] += 1.0;
                }
            }
        }

        {
            let mut comp = community_composition.as_array_mut();
            comp[[best_community, group]] += 1.0;
        }
        {
            let mut sizes = community_sizes.as_array_mut();
            sizes[best_community] += 1;
        }

        if (node_idx + 1) % 5000 == 0 {
            let pct = 100.0 * (node_idx + 1) as f64 / total_nodes as f64;
            println!(
                "Assigned {}/{} nodes ({:.1}%)",
                node_idx + 1,
                total_nodes,
                pct
            );
        }
    }

    let result = Array1::from(assignments);
    Ok(PyArray1::from_owned_array_bound(py, result))
}


/// Port of _run_edge_creation + establish_links while loop.
///
/// Processes all group pairs in one Rust call, maintaining the edge set
/// and adjacency list internally. Returns all new edges and final link counts.
#[pyfunction]
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None))]
fn run_edge_creation(
    // List of (src_id, dst_id, target_link_count) for each group pair
    group_pairs: Vec<(i64, i64, i64)>,
    // (src_id, dst_id) -> [community_ids] (may have duplicates for weighting)
    valid_communities_map: HashMap<(i64, i64), Vec<i64>>,
    // (src_id, dst_id) -> max link count
    maximum_num_links: HashMap<(i64, i64), i64>,
    // (community_id, group_id) -> [node_ids]
    communities_to_nodes: HashMap<(i64, i64), Vec<i64>>,
    // node_id -> group_id
    nodes_to_group: HashMap<i64, i64>,
    // Parameters
    fraction: f64,
    reciprocity_p: f64,
    transitivity_p: f64,
    pa_scope: String,
    number_of_communities: i64,
    bridge_probability: f64,
    // Optional pre-existing edges (for multiplex pre-seeding)
    pre_existing_edges: Option<Vec<(i64, i64)>>,
) -> PyResult<(Vec<(i64, i64)>, Vec<(i64, i64, i64)>)> {
    let mut rng = thread_rng();

    // Internal graph state
    let mut edges: HashSet<(i64, i64)> = HashSet::new();
    let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut new_edges: Vec<(i64, i64)> = Vec::new();

    // Popularity pools: (community_id, group_id) -> [node_ids]
    let mut popularity_pool: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    // Link counters
    let mut existing_num_links: HashMap<(i64, i64), i64> = HashMap::new();
    for &(src, dst) in maximum_num_links.keys() {
        existing_num_links.insert((src, dst), 0);
    }

    // Initialize internal state from pre-existing edges (multiplex pre-seeding)
    if let Some(ref pre_edges) = pre_existing_edges {
        for &(s, d) in pre_edges {
            edges.insert((s, d));
            adjacency.entry(s).or_default().push(d);
            // Count toward link budget (do NOT add to new_edges — already in graph)
            let s_group = *nodes_to_group.get(&s).unwrap_or(&-1);
            let d_group = *nodes_to_group.get(&d).unwrap_or(&-1);
            if s_group >= 0 && d_group >= 0 {
                *existing_num_links.entry((s_group, d_group)).or_insert(0) += 1;
            }
        }
        if !pre_edges.is_empty() {
            println!("  Rust: initialized with {} pre-existing edges", pre_edges.len());
        }
    }

    // Src node list cache per (community_id, group_id)
    let mut src_node_cache: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    let total_pairs = group_pairs.len();

    for (pair_idx, (src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        let src_id = *src_id;
        let dst_id = *dst_id;
        let target_link_count = *target_link_count;

        if (pair_idx + 1) % 500 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let possible_communities = match valid_communities_map.get(&(src_id, dst_id)) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };

        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);

        if num_links >= target_link_count {
            continue;
        }

        let max_attempts = target_link_count * 10;
        let mut attempts: i64 = 0;

        // Batch community selection
        let batch_size: usize = 10000;
        let pc_len = possible_communities.len();
        let mut community_batch: Vec<i64> = (0..batch_size)
            .map(|_| possible_communities[rng.gen_range(0..pc_len)])
            .collect();
        let mut batch_idx: usize = 0;

        while num_links < target_link_count && attempts < max_attempts {
            let community_id = community_batch[batch_idx];
            batch_idx += 1;

            if batch_idx >= batch_size {
                community_batch = (0..batch_size)
                    .map(|_| possible_communities[rng.gen_range(0..pc_len)])
                    .collect();
                batch_idx = 0;
            }

            // Get src nodes for this community
            let src_cache_key = (community_id, src_id);
            if !src_node_cache.contains_key(&src_cache_key) {
                let nodes = communities_to_nodes
                    .get(&src_cache_key)
                    .cloned()
                    .unwrap_or_default();
                src_node_cache.insert(src_cache_key, nodes);
            }
            let src_nodes = src_node_cache.get(&src_cache_key).unwrap();
            if src_nodes.is_empty() {
                attempts += 1;
                continue;
            }

            // Decide: bridge edge (dst from neighboring community) or normal
            let dst_community = if bridge_probability > 0.0
                && number_of_communities > 1
                && rng.gen::<f64>() < bridge_probability
            {
                let direction: i64 = if rng.gen::<bool>() { 1 } else { -1 };
                ((community_id + direction).rem_euclid(number_of_communities)) as i64
            } else {
                community_id
            };

            // Initialize popularity pool for (dst_community, dst_group)
            let pool_key = (dst_community, dst_id);
            if !popularity_pool.contains_key(&pool_key) {
                let dst_nodes = communities_to_nodes
                    .get(&pool_key)
                    .cloned()
                    .unwrap_or_default();
                if !dst_nodes.is_empty() {
                    let sample_size = ((dst_nodes.len() as f64) * fraction).ceil() as usize;
                    let sample_size = sample_size.min(dst_nodes.len());
                    let mut sampled = dst_nodes;
                    sampled.shuffle(&mut rng);
                    sampled.truncate(sample_size);
                    popularity_pool.insert(pool_key, sampled);
                } else {
                    popularity_pool.insert(pool_key, vec![]);
                }
            }

            let pool = popularity_pool.get(&pool_key).unwrap();
            if pool.is_empty() {
                attempts += 1;
                continue;
            }

            // Pick random src and dst
            let s = src_nodes[rng.gen_range(0..src_nodes.len())];
            let d = pool[rng.gen_range(0..pool.len())];

            if s != d && !edges.contains(&(s, d)) {
                edges.insert((s, d));
                adjacency.entry(s).or_default().push(d);
                new_edges.push((s, d));
                num_links += 1;
                existing_num_links.insert((src_id, dst_id), num_links);

                // Reciprocity
                if rng.gen::<f64>() < reciprocity_p {
                    let rev_existing = *existing_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    let rev_max = *maximum_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    if rev_existing < rev_max && !edges.contains(&(d, s)) {
                        edges.insert((d, s));
                        adjacency.entry(d).or_default().push(s);
                        new_edges.push((d, s));
                        *existing_num_links.entry((dst_id, src_id)).or_insert(0) += 1;
                        if dst_id == src_id {
                            num_links += 1;
                            existing_num_links.insert((src_id, dst_id), num_links);
                        }
                    }
                }

                // Preferential attachment
                if rng.gen::<f64>() > fraction && fraction != 1.0 {
                    if pa_scope == "global" {
                        for comm_id in 0..number_of_communities {
                            if rng.gen::<f64>() < (1.0 / number_of_communities as f64) * fraction {
                                let global_key = (comm_id, dst_id);
                                if let Some(p) = popularity_pool.get_mut(&global_key) {
                                    p.push(d);
                                }
                            }
                        }
                    } else {
                        if let Some(p) = popularity_pool.get_mut(&pool_key) {
                            p.push(d);
                        }
                    }
                }

                // Transitivity
                if transitivity_p >= rng.gen::<f64>() {
                    let neighbors: Vec<i64> = adjacency
                        .get(&d)
                        .cloned()
                        .unwrap_or_default();
                    for n in neighbors {
                        if s == n {
                            continue;
                        }
                        let n_id = match nodes_to_group.get(&n) {
                            Some(&id) => id,
                            None => continue,
                        };
                        let pair = (src_id, n_id);
                        let max_l = match maximum_num_links.get(&pair) {
                            Some(&v) => v,
                            None => continue,
                        };
                        let existing = *existing_num_links.get(&pair).unwrap_or(&0);
                        if existing < max_l && !edges.contains(&(s, n)) {
                            edges.insert((s, n));
                            adjacency.entry(s).or_default().push(n);
                            new_edges.push((s, n));
                            *existing_num_links.entry(pair).or_insert(0) += 1;

                            if n_id == dst_id {
                                num_links += 1;
                                existing_num_links.insert((src_id, dst_id), num_links);
                            }

                            // Reciprocity for transitive edge
                            if rng.gen::<f64>() < reciprocity_p {
                                let rev_pair = (n_id, src_id);
                                let rev_existing =
                                    *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                let rev_max =
                                    *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                    edges.insert((n, s));
                                    adjacency.entry(n).or_default().push(s);
                                    new_edges.push((n, s));
                                    *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                    if n_id == src_id && src_id == dst_id {
                                        num_links += 1;
                                        existing_num_links
                                            .insert((src_id, dst_id), num_links);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            attempts += 1;
        }
    }

    // Convert existing_num_links to flat triples
    let links_out: Vec<(i64, i64, i64)> = existing_num_links
        .into_iter()
        .map(|((s, d), c)| (s, d, c))
        .collect();

    Ok((new_edges, links_out))
}


/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    Ok(())
}
