use std::collections::{HashMap, HashSet};

use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2} ;
#[pyfunction]
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None, node_coordinates=None, internal_transitivity_p=-1.0, external_transitivity_p=-1.0))]
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
    // Optional node coordinates for Phase B spatial ring search
    node_coordinates: Option<HashMap<i64, f64>>,
    // Community-keyed transitivity. A closure s→n (through pivot d) is INTERNAL
    // when n and d share the same community id (group ignored — option 2),
    // EXTERNAL otherwise. -1.0 on either side falls back to the scalar
    // transitivity_p, so existing call sites keep working.
    internal_transitivity_p: f64,
    external_transitivity_p: f64,
) -> PyResult<(Vec<(i64, i64)>, Vec<(i64, i64, i64)>)> {
    let mut rng = thread_rng();

    // Effective per-side transitivity (fall back to scalar when negative).
    let int_trans_p = if internal_transitivity_p < 0.0 { transitivity_p } else { internal_transitivity_p };
    let ext_trans_p = if external_transitivity_p < 0.0 { transitivity_p } else { external_transitivity_p };
    let use_transitivity = int_trans_p > 0.0 || ext_trans_p > 0.0;

    // ── Phase diagnostics counters ────────────────────────────────────────
    let mut phase_a_primary: u64 = 0;
    let mut phase_a_reciprocity: u64 = 0;
    let mut phase_a_trans_int: u64 = 0;
    let mut phase_a_trans_ext: u64 = 0;
    let mut phase_a_pairs_touched: u64 = 0;
    let mut phase_b_primary: u64 = 0;
    let mut phase_b_reciprocity: u64 = 0;
    let mut phase_b_trans_int: u64 = 0;
    let mut phase_b_trans_ext: u64 = 0;
    let mut phase_b_pairs_touched: u64 = 0;

    // Internal graph state
    let mut edges: HashSet<(i64, i64)> = HashSet::new();
    let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
    // Incoming edges, so the transitivity scan can use the UNDIRECTED
    // neighbourhood out[d] ∪ in[d] (matches NetworkX's neighbors(d)). Only
    // populated when transitivity is active, so the baseline pays nothing.
    let mut in_adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut new_edges: Vec<(i64, i64)> = Vec::new();

    // Popularity pools: (community_id, group_id) -> [node_ids]
    let mut popularity_pool: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    // Link counters
    let mut existing_num_links: HashMap<(i64, i64), i64> = HashMap::new();
    for &(src, dst) in maximum_num_links.keys() {
        existing_num_links.insert((src, dst), 0);
    }

    // Per-node community lookup (community id is the FIRST tuple element).
    // node -> (community_id, group_id); the gate compares only .0 (community id).
    let mut node_to_community: HashMap<i64, (i64, i64)> = HashMap::new();
    if use_transitivity {
        for (&comm_key, nodes) in &communities_to_nodes {
            for &nn in nodes {
                node_to_community.insert(nn, comm_key);
            }
        }
    }

    // Initialize internal state from pre-existing edges (multiplex pre-seeding)
    if let Some(ref pre_edges) = pre_existing_edges {
        for &(s, d) in pre_edges {
            edges.insert((s, d));
            adjacency.entry(s).or_default().push(d);
            if use_transitivity { in_adjacency.entry(d).or_default().push(s); }
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

    // Phase B: src nodes sorted by coordinate, dst communities sorted by centroid.
    // Ring search finds nearest communities then picks a random node from each —
    // spreading degree load across all nodes rather than targeting edge-nearest ones.
    let mut group_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    let mut group_comm_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    if let Some(ref nc) = node_coordinates {
        let mut group_all_nodes: HashMap<i64, Vec<i64>> = HashMap::new();
        for (&(_, gid), nodes) in &communities_to_nodes {
            group_all_nodes.entry(gid).or_default().extend(nodes.iter().copied());
        }
        for (gid, nodes) in &group_all_nodes {
            let mut sorted: Vec<(f64, i64)> = nodes.iter()
                .map(|&n| (*nc.get(&n).unwrap_or(&0.5), n))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            group_sorted.insert(*gid, sorted);
        }
        // Community centroids: average coordinate of nodes in (comm, group)
        for (&(comm_id, gid), nodes) in &communities_to_nodes {
            if nodes.is_empty() { continue; }
            let centroid: f64 = nodes.iter()
                .map(|&n| *nc.get(&n).unwrap_or(&0.5))
                .sum::<f64>() / nodes.len() as f64;
            group_comm_sorted.entry(gid).or_default().push((centroid, comm_id));
        }
        for sorted in group_comm_sorted.values_mut() {
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        println!("  Phase B: built sorted arrays for {} groups", group_sorted.len());
    }

    let total_pairs = group_pairs.len();
    let pre_edge_count = pre_existing_edges.as_ref().map_or(0, |v| v.len());

    for (pair_idx, (src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        let src_id = *src_id;
        let dst_id = *dst_id;
        let target_link_count = *target_link_count;

        if (pair_idx + 1) % 5000 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let possible_communities = valid_communities_map
            .get(&(src_id, dst_id))
            .filter(|v| !v.is_empty());

        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);

        if num_links >= target_link_count {
            continue;
        }

        let edges_at_pair_start = new_edges.len();

        // ── Phase A: community-based edge creation ────────────────────────────
        // Communities are iterated sequentially (shuffled once per pair) so each
        // community exhausts a proportional quota before moving to the next.
        // This concentrates edges within communities, raising transitivity.
        if let Some(communities) = possible_communities {
            let mut comm_order: Vec<i64> = {
                let mut seen = std::collections::HashSet::new();
                communities.iter().filter(|&&c| seen.insert(c)).cloned().collect()
            };
            comm_order.shuffle(&mut rng);
            let n_comms = comm_order.len();

            let max_passes: i64 = 3;
            let mut pass: i64 = 0;

            'outer: while num_links < target_link_count && pass < max_passes {
                pass += 1;
                for &community_id in &comm_order {
                    if num_links >= target_link_count { break 'outer; }

                    let remaining = (target_link_count - num_links) as usize;
                    let quota = ((remaining + n_comms - 1) / n_comms).max(1);

                    // Get src nodes for this community
                    let src_cache_key = (community_id, src_id);
                    if !src_node_cache.contains_key(&src_cache_key) {
                        let nodes = communities_to_nodes
                            .get(&src_cache_key)
                            .cloned()
                            .unwrap_or_default();
                        src_node_cache.insert(src_cache_key, nodes);
                    }
                    if src_node_cache.get(&src_cache_key).unwrap().is_empty() {
                        continue;
                    }

                    // Bridge or normal dst community
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
                    if popularity_pool.get(&pool_key).unwrap().is_empty() {
                        continue;
                    }

                    // Create up to quota edges within this community
                    let mut created = 0usize;
                    let mut local_attempts = 0usize;
                    let max_local = quota * 3;

                    while created < quota && local_attempts < max_local && num_links < target_link_count {
                        local_attempts += 1;

                        let s = {
                            let src_nodes = src_node_cache.get(&src_cache_key).unwrap();
                            src_nodes[rng.gen_range(0..src_nodes.len())]
                        };
                        let d = {
                            let pool = popularity_pool.get(&pool_key).unwrap();
                            pool[rng.gen_range(0..pool.len())]
                        };

                        if s != d && !edges.contains(&(s, d)) {
                            edges.insert((s, d));
                            adjacency.entry(s).or_default().push(d);
                            if use_transitivity { in_adjacency.entry(d).or_default().push(s); }
                            new_edges.push((s, d));
                            phase_a_primary += 1;
                            num_links += 1;
                            existing_num_links.insert((src_id, dst_id), num_links);
                            created += 1;

                            // Reciprocity
                            if rng.gen::<f64>() < reciprocity_p {
                                let rev_existing = *existing_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                                let rev_max = *maximum_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                                if rev_existing < rev_max && !edges.contains(&(d, s)) {
                                    edges.insert((d, s));
                                    adjacency.entry(d).or_default().push(s);
                                    if use_transitivity { in_adjacency.entry(s).or_default().push(d); }
                                    new_edges.push((d, s));
                                    phase_a_reciprocity += 1;
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
                                    if rng.gen::<f64>() > fraction {
                                        if let Some(p) = popularity_pool.get_mut(&pool_key) {
                                            p.push(d);
                                            if let Some(dst_community_nodes) = communities_to_nodes.get(&pool_key) {
                                                if !dst_community_nodes.is_empty() {
                                                    let r = rng.gen_range(0..dst_community_nodes.len());
                                                    p.push(dst_community_nodes[r]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // ── Transitivity (community-keyed, undirected scan) ──
                            // Per-neighbour roll: int_trans_p when n shares d's
                            // community id, ext_trans_p otherwise. Target check
                            // breaks the loop once this pair is full.
                            if use_transitivity {
                                // Undirected neighbourhood of d, deduplicated.
                                let mut neighbors: Vec<i64> = adjacency.get(&d).cloned().unwrap_or_default();
                                if let Some(ins) = in_adjacency.get(&d) {
                                    neighbors.extend(ins.iter().copied());
                                }
                                {
                                    let mut seen = std::collections::HashSet::new();
                                    neighbors.retain(|&x| seen.insert(x));
                                }
                                let d_comm = node_to_community.get(&d).copied();

                                for n in neighbors {
                                    if num_links >= target_link_count { break; }
                                    if s == n { continue; }

                                    // Internal iff same community id (group ignored);
                                    // two unknowns are NOT treated as equal.
                                    let n_comm = node_to_community.get(&n).copied();
                                    let internal = match (n_comm, d_comm) {
                                        (Some(nc), Some(dc)) => nc.0 == dc.0,
                                        _ => false,
                                    };
                                    let closure_p = if internal { int_trans_p } else { ext_trans_p };
                                    if rng.gen::<f64>() >= closure_p { continue; }

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
                                        in_adjacency.entry(n).or_default().push(s);
                                        new_edges.push((s, n));
                                        if internal { phase_a_trans_int += 1; } else { phase_a_trans_ext += 1; }
                                        *existing_num_links.entry(pair).or_insert(0) += 1;
                                        if n_id == dst_id {
                                            num_links += 1;
                                            existing_num_links.insert((src_id, dst_id), num_links);
                                        }
                                        // Reciprocity for transitive edge
                                        if rng.gen::<f64>() < reciprocity_p {
                                            let rev_pair = (n_id, src_id);
                                            let rev_existing = *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                            let rev_max = *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                            if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                                edges.insert((n, s));
                                                adjacency.entry(n).or_default().push(s);
                                                in_adjacency.entry(s).or_default().push(n);
                                                new_edges.push((n, s));
                                                phase_a_reciprocity += 1;
                                                *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                                if n_id == src_id && src_id == dst_id {
                                                    num_links += 1;
                                                    existing_num_links.insert((src_id, dst_id), num_links);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // end Phase A

        if new_edges.len() > edges_at_pair_start {
            phase_a_pairs_touched += 1;
        }
    } // end pair loop

    // ── Phase B: spatial ring search for remaining budget ────────────────────
    // For each pair still under budget, finds nearest dst communities by
    // centroid and picks a random node — fills cross-block pairs left by A/B.
    for (pair_idx, &(src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        if (pair_idx + 1) % 5000 == 0 {
            println!("Phase B: pair {} of {}", pair_idx + 1, total_pairs);
        }
        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);
        if num_links >= target_link_count { continue; }

        let edges_at_pair_start = new_edges.len();

        if let (Some(src_sorted), Some(dst_comm_sorted)) =
            (group_sorted.get(&src_id), group_comm_sorted.get(&dst_id))
        {
            const PHASE_C_COMM_WINDOW: usize = 200;
            let n_dst_comm = dst_comm_sorted.len();
            let win = PHASE_C_COMM_WINDOW.min(n_dst_comm);
            let n_src = src_sorted.len();

            if n_src > 0 && win > 0 {
                let mut src_indices: Vec<usize> = (0..n_src).collect();
                loop {
                    if num_links >= target_link_count { break; }
                    src_indices.shuffle(&mut rng);
                    let remaining = (target_link_count - num_links) as usize;
                    let edges_per_src = ((remaining + n_src - 1) / n_src).max(1).min(win);
                    let mut made_progress = false;

                    for &si in &src_indices {
                        if num_links >= target_link_count { break; }
                        let (theta_s, s) = src_sorted[si];
                        let center = dst_comm_sorted.partition_point(|&(c, _)| c < theta_s);
                        let mut found = 0usize;

                        'delta: for delta in 0..win {
                            if found >= edges_per_src { break 'delta; }
                            let j1 = (center + delta) % n_dst_comm;
                            let j2 = (center + n_dst_comm - delta - 1) % n_dst_comm;
                            for &j in &[j1, j2] {
                                if found >= edges_per_src || num_links >= target_link_count { break; }
                                let (_, comm_id) = dst_comm_sorted[j];
                                let pool_key = (comm_id, dst_id);
                                if let Some(dst_nodes) = communities_to_nodes.get(&pool_key) {
                                    if !dst_nodes.is_empty() {
                                        let d = dst_nodes[rng.gen_range(0..dst_nodes.len())];
                                        if s != d && !edges.contains(&(s, d)) {
                                            edges.insert((s, d));
                                            adjacency.entry(s).or_default().push(d);
                                            if use_transitivity { in_adjacency.entry(d).or_default().push(s); }
                                            new_edges.push((s, d));
                                            phase_b_primary += 1;
                                            *existing_num_links.entry((src_id, dst_id)).or_insert(0) += 1;
                                            num_links += 1;
                                            found += 1;
                                            made_progress = true;

                                            // Reciprocity (same logic as Phase A)
                                            if rng.gen::<f64>() < reciprocity_p {
                                                let rev_existing = *existing_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                                                let rev_max = *maximum_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                                                if rev_existing < rev_max && !edges.contains(&(d, s)) {
                                                    edges.insert((d, s));
                                                    adjacency.entry(d).or_default().push(s);
                                                    if use_transitivity { in_adjacency.entry(s).or_default().push(d); }
                                                    new_edges.push((d, s));
                                                    phase_b_reciprocity += 1;
                                                    *existing_num_links.entry((dst_id, src_id)).or_insert(0) += 1;
                                                    if dst_id == src_id {
                                                        num_links += 1;
                                                        existing_num_links.insert((src_id, dst_id), num_links);
                                                    }
                                                }
                                            }

                                            // ── Transitivity (community-keyed, undirected scan) ──
                                            if use_transitivity {
                                                let mut neighbors: Vec<i64> = adjacency.get(&d).cloned().unwrap_or_default();
                                                if let Some(ins) = in_adjacency.get(&d) {
                                                    neighbors.extend(ins.iter().copied());
                                                }
                                                {
                                                    let mut seen = std::collections::HashSet::new();
                                                    neighbors.retain(|&x| seen.insert(x));
                                                }
                                                let d_comm = node_to_community.get(&d).copied();

                                                for n in neighbors {
                                                    if num_links >= target_link_count { break; }
                                                    if s == n { continue; }

                                                    let n_comm = node_to_community.get(&n).copied();
                                                    let internal = match (n_comm, d_comm) {
                                                        (Some(nc), Some(dc)) => nc.0 == dc.0,
                                                        _ => false,
                                                    };
                                                    let closure_p = if internal { int_trans_p } else { ext_trans_p };
                                                    if rng.gen::<f64>() >= closure_p { continue; }

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
                                                        in_adjacency.entry(n).or_default().push(s);
                                                        new_edges.push((s, n));
                                                        if internal { phase_b_trans_int += 1; } else { phase_b_trans_ext += 1; }
                                                        *existing_num_links.entry(pair).or_insert(0) += 1;
                                                        if n_id == dst_id {
                                                            num_links += 1;
                                                            existing_num_links.insert((src_id, dst_id), num_links);
                                                        }
                                                        // Reciprocity for transitive edge
                                                        if rng.gen::<f64>() < reciprocity_p {
                                                            let rev_pair = (n_id, src_id);
                                                            let rev_existing = *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                                            let rev_max = *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                                            if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                                                edges.insert((n, s));
                                                                adjacency.entry(n).or_default().push(s);
                                                                in_adjacency.entry(s).or_default().push(n);
                                                                new_edges.push((n, s));
                                                                phase_b_reciprocity += 1;
                                                                *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                                                if n_id == src_id && src_id == dst_id {
                                                                    num_links += 1;
                                                                    existing_num_links.insert((src_id, dst_id), num_links);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !made_progress { break; }
                }
            }
        }

        if new_edges.len() > edges_at_pair_start {
            phase_b_pairs_touched += 1;
        }
    } // end Phase B

    // ── Phase diagnostics report ──────────────────────────────────────────
    let phase_a_trans = phase_a_trans_int + phase_a_trans_ext;
    let phase_b_trans = phase_b_trans_int + phase_b_trans_ext;
    let phase_a_total = phase_a_primary + phase_a_reciprocity + phase_a_trans;
    let phase_b_total = phase_b_primary + phase_b_reciprocity + phase_b_trans;
    let grand_total = phase_a_total + phase_b_total;
    let pct = |x: u64| -> f64 {
        if grand_total == 0 { 0.0 } else { 100.0 * x as f64 / grand_total as f64 }
    };

    println!("\n┌─ Edge creation diagnostics ─────────────────────────────");
    println!("│ Phase A (community-based)");
    println!("│   primary         : {:>10}   ({:>5.1}%)", phase_a_primary,     pct(phase_a_primary));
    println!("│   reciprocity      : {:>10}   ({:>5.1}%)", phase_a_reciprocity, pct(phase_a_reciprocity));
    println!("│   transitivity int : {:>10}   ({:>5.1}%)", phase_a_trans_int,   pct(phase_a_trans_int));
    println!("│   transitivity ext : {:>10}   ({:>5.1}%)", phase_a_trans_ext,   pct(phase_a_trans_ext));
    println!("│   subtotal         : {:>10}   ({:>5.1}%)", phase_a_total,       pct(phase_a_total));
    println!("│   pairs touched    : {:>10} / {}", phase_a_pairs_touched, total_pairs);
    println!("│ Phase B (ring search)");
    println!("│   primary         : {:>10}   ({:>5.1}%)", phase_b_primary,     pct(phase_b_primary));
    println!("│   reciprocity      : {:>10}   ({:>5.1}%)", phase_b_reciprocity, pct(phase_b_reciprocity));
    println!("│   transitivity int : {:>10}   ({:>5.1}%)", phase_b_trans_int,   pct(phase_b_trans_int));
    println!("│   transitivity ext : {:>10}   ({:>5.1}%)", phase_b_trans_ext,   pct(phase_b_trans_ext));
    println!("│   subtotal         : {:>10}   ({:>5.1}%)", phase_b_total,       pct(phase_b_total));
    println!("│   pairs touched    : {:>10} / {}", phase_b_pairs_touched, total_pairs);
    println!("│ Transitivity gate : community-id  (int_p={:.2}, ext_p={:.2})", int_trans_p, ext_trans_p);
    println!("│ Total new edges   : {:>10}", grand_total);
    let counted_plus_pre = grand_total as usize + pre_edge_count;
    let actual = new_edges.len() + pre_edge_count;
    if counted_plus_pre != actual {
        println!("│ ⚠ counter mismatch: counted={} actual={} (diff={})",
            counted_plus_pre, actual, actual as i64 - counted_plus_pre as i64);
    }
    println!("└─────────────────────────────────────────────────────────");

    // Convert existing_num_links to flat triples
    let links_out: Vec<(i64, i64, i64)> = existing_num_links
        .into_iter()
        .map(|((s, d), c)| (s, d, c))
        .collect();

    Ok((new_edges, links_out))
}

/// Fast O(N) community assignment — no SA, uniform group distribution.
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, budget, n_groups, initial_num_communities, target_counts=None, total_nodes=0, new_comm_penalty=3.0, initial_comp=None))]
fn process_nodes_capacity_fast<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    initial_num_communities: usize,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
    new_comm_penalty: f64,
    initial_comp: Option<HashMap<usize, HashMap<usize, i64>>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (all_nodes, budget, new_comm_penalty, initial_comp);
    let node_groups = node_groups.as_array();
    let k = initial_num_communities.max(1);
    let mut rng = thread_rng();

    let tc: Option<Vec<i32>> = target_counts.map(|t| t.as_array().to_owned().to_vec());

    let mut group_counts = vec![0usize; n_groups];
    for i in 0..total_nodes {
        let g = node_groups[i] as usize;
        if g < n_groups {
            group_counts[g] += 1;
        }
    }

    let group_lists: Vec<Vec<usize>> = (0..n_groups)
        .map(|g| {
            let p = group_counts[g];
            if p == 0 { return Vec::new(); }
            let mut list = Vec::with_capacity(p);

            if let Some(ref tc_arr) = tc {
                let total_target: i64 = tc_arr.iter().map(|&x| x as i64).sum();
                let denom = if total_target > 0 { total_target as f64 } else { k as f64 };
                let mut assigned = 0usize;
                for c in 0..k {
                    let share = if c < k - 1 {
                        let raw = (p as f64 * tc_arr[c] as f64 / denom).round() as usize;
                        raw.min(p - assigned)
                    } else {
                        p - assigned
                    };
                    for _ in 0..share { list.push(c); }
                    assigned += share;
                }
            } else {
                let mut comm_order: Vec<usize> = (0..k).collect();
                comm_order.shuffle(&mut rng);
                let base = p / k;
                let remainder = p % k;
                for (i, &c) in comm_order.iter().enumerate() {
                    let count = base + if i < remainder { 1 } else { 0 };
                    for _ in 0..count { list.push(c); }
                }
            }

            list.shuffle(&mut rng);
            list
        })
        .collect();

    let mut group_cursors = vec![0usize; n_groups];
    let mut assignments = Vec::with_capacity(total_nodes);

    for i in 0..total_nodes {
        let g = node_groups[i] as usize;
        let comm = if g < n_groups && group_cursors[g] < group_lists[g].len() {
            let c = group_lists[g][group_cursors[g]];
            group_cursors[g] += 1;
            c
        } else {
            rng.gen_range(0..k)
        };
        assignments.push(comm as i64);
    }

    println!("Fast assignment complete: {} nodes -> {} communities", total_nodes, k);
    Ok(PyArray1::from_owned_array_bound(py, Array1::from(assignments)))
}

#[pyfunction]
#[pyo3(signature = (
    assignments, node_groups, budget,
    n_groups, n_communities,
    n_iterations = 100_000,
    overshoot_penalty = 10.0,
    seed = 42,
))]
fn refine_communities_move<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    mut n_communities: usize, // Made mutable to allow growth
    n_iterations: usize,
    overshoot_penalty: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let assigns = assignments.as_array();
    let groups = node_groups.as_array();
    let n = assigns.len();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut current: Vec<usize> = (0..n).map(|i| assigns[i] as usize).collect();

    // comp[c] = {group: count}. Using a Vec of HashMaps.
    let mut comp: Vec<HashMap<usize, i64>> = (0..n_communities).map(|_| HashMap::new()).collect();
    for i in 0..n {
        let c = current[i];
        let g = groups[i] as usize;
        if c < n_communities && g < n_groups {
            *comp[c].entry(g).or_insert(0) += 1;
        }
    }

    let mut achieved: HashMap<(usize, usize), i64> = HashMap::new();
    for c_comp in &comp {
        for (&g, &cg) in c_comp {
            for (&h, &ch) in c_comp {
                *achieved.entry((g, h)).or_insert(0) += cg * ch;
            }
        }
    }

    let pair_cost = |achieved_val: i64, budget_val: i64| -> f64 {
        let d = achieved_val as f64 - budget_val as f64;
        if d > 0.0 { overshoot_penalty * d } else { -d }
    };

    let mut current_loss = 0.0;
    // Initial loss calculation
    for (&(g, h), &av) in &achieved {
        let bv = budget.get(&(g as i64, h as i64)).copied().unwrap_or(0);
        current_loss += pair_cost(av, bv);
    }
    for (&(g, h), &bv) in &budget {
        if !achieved.contains_key(&(g as usize, h as usize)) {
            current_loss += pair_cost(0, bv);
        }
    }

    let mut accepted = 0usize;
    let report_every = (n_iterations / 10).max(1);

    for iter in 0..n_iterations {
        let i = rng.gen_range(0..n);
        let g = groups[i] as usize;
        let c_old = current[i];
        
        // Pick a target community: existing OR one brand new potential ID
        let c_new = rng.gen_range(0..=n_communities);

        if c_old == c_new { continue; }

        // If we chose a new community, ensure the comp vector can hold it
        if c_new == n_communities {
            // We don't actually push to comp yet, just prepare the logic
        }

        let mut delta_loss = 0.0;
        let mut affected_updates: Vec<((usize, usize), i64)> = Vec::new();

        // The only group whose count changes is 'g'.
        // This affects pairs (g, h) and (h, g) for all h present in c_old or c_new.
        let mut affected_groups: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &h in comp[c_old].keys() { affected_groups.insert(h); }
        if c_new < n_communities {
            for &h in comp[c_new].keys() { affected_groups.insert(h); }
        }
        affected_groups.insert(g); 

        for &h in &affected_groups {
            let c_old_g_old = *comp[c_old].get(&g).unwrap_or(&0);
            let c_old_h_old = *comp[c_old].get(&h).unwrap_or(&0);
            
            let c_new_g_old = if c_new < n_communities { *comp[c_new].get(&g).unwrap_or(&0) } else { 0 };
            let c_new_h_old = if c_new < n_communities { *comp[c_new].get(&h).unwrap_or(&0) } else { 0 };

            // After move: c_old[g] decreases, c_new[g] increases
            let c_old_g_new = c_old_g_old - 1;
            let c_new_g_new = c_new_g_old + 1;
            
            // h doesn't change, but if h == g, we use the new values
            let c_old_h_new = if h == g { c_old_g_new } else { c_old_h_old };
            let c_new_h_new = if h == g { c_new_g_new } else { c_new_h_old };

            let old_contrib = (c_old_g_old * c_old_h_old) + (c_new_g_old * c_new_h_old);
            let new_contrib = (c_old_g_new * c_old_h_new) + (c_new_g_new * c_new_h_new);
            let delta = new_contrib - old_contrib;

            if delta != 0 {
                let keys = if g == h { vec![(g, g)] } else { vec![(g, h), (h, g)] };
                for (ga, ha) in keys {
                    let av_old = *achieved.get(&(ga, ha)).unwrap_or(&0);
                    let bv = budget.get(&(ga as i64, ha as i64)).copied().unwrap_or(0);
                    let av_new = av_old + delta;
                    delta_loss += pair_cost(av_new, bv) - pair_cost(av_old, bv);
                    affected_updates.push(((ga, ha), delta));
                }
            }
        }

        if delta_loss < 0.0 {
            // Apply Move
            if c_new == n_communities {
                comp.push(HashMap::new());
                n_communities += 1;
            }

            // Update comp
            let count_old = comp[c_old].get_mut(&g).unwrap();
            *count_old -= 1;
            if *count_old == 0 { comp[c_old].remove(&g); }
            *comp[c_new].entry(g).or_insert(0) += 1;

            // Update achieved
            for ((ga, ha), d) in affected_updates {
                let entry = achieved.entry((ga, ha)).or_insert(0);
                *entry += d;
                if *entry == 0 { achieved.remove(&(ga, ha)); }
            }

            current[i] = c_new;
            current_loss += delta_loss;
            accepted += 1;
        }

        if (iter + 1) % report_every == 0 {
            println!("Iter {}: loss={:.2}, communities={}", iter+1, current_loss, n_communities);
        }
    }

    let result: Vec<i64> = current.iter().map(|&c| c as i64).collect();
    Ok(PyArray1::from_owned_array_bound(py, Array1::from(result)))
}


/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity_fast, m)?)?;
    m.add_function(wrap_pyfunction!(refine_communities_move, m)?)?;
    Ok(())
}