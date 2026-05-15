// Bisection harness: runs the kokoro-v1-debug model (which exposes intermediate
// tensors) and compares per-tensor stats against ORT (bisect_taps.json).

extern crate alloc;

use burn::prelude::*;
use serde::Deserialize;
use std::fs;

model_checks_common::backend_type!();

include!(concat!(env!("OUT_DIR"), "/model/kokoro-v1-debug.rs"));

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Tap {
    op_type: String,
    tensor: String,
    shape: Vec<usize>,
    dtype: String,
    abs_max: f64,
    abs_mean: f64,
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

#[derive(Debug, Deserialize)]
struct Taps {
    taps: Vec<Tap>,
}

fn compare(ort_index: usize, ort: &Tap, burn: &[f32]) -> (f64, f64) {
    if burn.is_empty() {
        println!("  [{:>3}] EMPTY ({})", ort_index, ort.tensor);
        return (0.0, 0.0);
    }
    let mut amax = 0.0f32;
    let mut abs_sum = 0.0f64;
    for &v in burn {
        let av = v.abs();
        abs_sum += av as f64;
        if av > amax {
            amax = av;
        }
    }
    let n = burn.len() as f64;
    let babs_mean = abs_sum / n;
    let abs_max_diff = (amax as f64 - ort.abs_max).abs();
    let abs_mean_diff = (babs_mean - ort.abs_mean).abs();
    let rel_max = abs_max_diff / ort.abs_max.max(1e-6);
    let rel_mean = abs_mean_diff / ort.abs_mean.max(1e-6);

    let marker = if rel_max > 0.10 || rel_mean > 0.10 {
        "  XX"
    } else if rel_max > 0.001 || rel_mean > 0.001 {
        "  ~"
    } else {
        "    "
    };
    let tail = if ort.tensor.len() > 60 {
        format!("...{}", &ort.tensor[ort.tensor.len() - 57..])
    } else {
        ort.tensor.clone()
    };
    println!(
        "{} [{:>3}] {:14} burn[abs_max={:>11.4} abs_mean={:>9.4}]  ort[abs_max={:>11.4} abs_mean={:>9.4}]  Δrel={:>7.3}  {}",
        marker, ort_index, ort.op_type, amax, babs_mean, ort.abs_max, ort.abs_mean, rel_max, tail
    );
    (rel_max, rel_mean)
}

fn main() {
    let artifacts = model_checks_common::artifacts_dir("kokoro");

    let taps: Taps = serde_json::from_str(
        &fs::read_to_string(artifacts.join("bisect_taps.json")).expect("read bisect_taps.json"),
    )
    .expect("parse bisect_taps.json");
    println!("Loaded {} reference taps", taps.taps.len());

    let device = model_checks_common::best_device!();
    println!("Loading kokoro-v1-debug...");
    let weights = concat!(env!("OUT_DIR"), "/model/kokoro-v1-debug.bpk");
    let model: Model = Model::from_file(weights, &device);

    #[derive(Deserialize)]
    struct RefIn {
        tokens: Vec<i64>,
        style: Vec<f32>,
        speed: f32,
    }
    let r: RefIn = serde_json::from_str(
        &fs::read_to_string(artifacts.join("reference_outputs.json"))
            .expect("read reference_outputs.json"),
    )
    .expect("parse reference_outputs.json");

    let seq_len = r.tokens.len();
    let tokens = Tensor::<1, Int>::from_ints(r.tokens.as_slice(), &device).reshape([1, seq_len]);
    let style = Tensor::<1>::from_floats(r.style.as_slice(), &device).reshape([1, 256]);
    let speed = Tensor::<1>::from_floats([r.speed].as_slice(), &device);

    println!("Running burn forward (debug build)...");
    let out = model.forward(tokens, style, speed);

    macro_rules! v {
        ($t:expr) => {
            $t.to_data().convert::<f32>().to_vec().unwrap()
        };
    }

    // Mapping: Burn[0] = audio (graph_output, ORT[N-1]); Burn[i+1] = ORT[i].
    let burn_taps: Vec<Vec<f32>> = vec![
        v!(out.1),
        v!(out.2),
        v!(out.3),
        v!(out.4),
        v!(out.5),
        v!(out.6),
        v!(out.7),
        v!(out.8),
        v!(out.9),
        v!(out.10),
        v!(out.11),
        v!(out.12),
        v!(out.13),
        v!(out.14),
        v!(out.15),
        v!(out.16),
        v!(out.17),
        v!(out.18),
        v!(out.19),
        v!(out.20),
        v!(out.21),
        v!(out.22),
        v!(out.23),
        v!(out.24),
        v!(out.25),
        v!(out.26),
        v!(out.27),
        v!(out.28),
        v!(out.29),
        v!(out.30),
        v!(out.31),
        v!(out.32),
        v!(out.33),
        v!(out.34),
        v!(out.35),
        v!(out.36),
        v!(out.37),
        v!(out.38),
        v!(out.39),
        v!(out.40),
        v!(out.41),
        v!(out.42),
        v!(out.43),
        v!(out.44),
        v!(out.45),
        v!(out.46),
        v!(out.47),
        v!(out.48),
        v!(out.49),
        v!(out.50),
        v!(out.51),
        v!(out.52),
        v!(out.53),
        v!(out.54),
        v!(out.55),
        v!(out.56),
        v!(out.57),
        v!(out.58),
        v!(out.59),
        v!(out.60),
        v!(out.61),
        v!(out.62),
        v!(out.63),
        v!(out.64),
        v!(out.65),
        v!(out.66),
        v!(out.67),
        v!(out.68),
        v!(out.69),
        v!(out.70),
        v!(out.71),
        v!(out.72),
        v!(out.73),
        v!(out.74),
        v!(out.75),
        v!(out.76),
        v!(out.77),
        v!(out.78),
        v!(out.79),
        v!(out.80),
        v!(out.81),
        v!(out.82),
        v!(out.83),
        v!(out.84),
        v!(out.85),
        v!(out.86),
        v!(out.87),
        v!(out.88),
        v!(out.89),
        v!(out.90),
        v!(out.91),
        v!(out.92),
        v!(out.93),
        v!(out.94),
        v!(out.95),
        v!(out.96),
        v!(out.97),
        v!(out.98),
        v!(out.99),
        v!(out.100),
        v!(out.101),
        v!(out.102),
        v!(out.103),
        v!(out.104),
        v!(out.105),
        v!(out.106),
        v!(out.107),
        v!(out.108),
        v!(out.109),
        v!(out.110),
        v!(out.111),
        v!(out.112),
        v!(out.113),
        v!(out.114),
        v!(out.115),
        v!(out.116),
        v!(out.117),
        v!(out.118),
        v!(out.119),
        v!(out.120),
        v!(out.121),
        v!(out.122),
        v!(out.123),
        v!(out.124),
        v!(out.125),
        v!(out.126),
        v!(out.127),
        v!(out.128),
        v!(out.129),
        v!(out.130),
        v!(out.131),
        v!(out.132),
        v!(out.133),
        v!(out.134),
        v!(out.135),
        v!(out.136),
        v!(out.137),
        v!(out.138),
        v!(out.139),
        v!(out.140),
        v!(out.141),
        v!(out.142),
        v!(out.143),
        v!(out.144),
        v!(out.145),
        v!(out.146),
        v!(out.147),
        v!(out.148),
        v!(out.149),
        v!(out.150),
        v!(out.151),
        v!(out.152),
        v!(out.153),
        v!(out.154),
        v!(out.155),
        v!(out.156),
        v!(out.157),
        v!(out.158),
        v!(out.159),
        v!(out.160),
        v!(out.161),
        v!(out.162),
        v!(out.163),
        v!(out.164),
        v!(out.165),
        v!(out.166),
        v!(out.167),
        v!(out.168),
        v!(out.169),
        v!(out.170),
        v!(out.171),
        v!(out.172),
        v!(out.173),
        v!(out.174),
        v!(out.175),
        v!(out.176),
        v!(out.177),
        v!(out.178),
        v!(out.179),
        v!(out.180),
        v!(out.181),
        v!(out.182),
        v!(out.183),
        v!(out.184),
        v!(out.185),
        v!(out.186),
        v!(out.187),
        v!(out.188),
        v!(out.189),
        v!(out.190),
        v!(out.191),
        v!(out.192),
        v!(out.193),
        v!(out.194),
        v!(out.195),
        v!(out.196),
    ];
    let burn_audio = v!(out.0);

    println!("\nPer-tap divergence (XX = >10%, ~ = >0.1%):");
    let n_taps = taps.taps.len();
    let mut first_diverge: Option<usize> = None;
    for i in 0..(n_taps - 1) {
        let (rel_max, rel_mean) = compare(i, &taps.taps[i], &burn_taps[i]);
        if first_diverge.is_none() && (rel_max > 0.001 || rel_mean > 0.001) {
            first_diverge = Some(i);
        }
    }
    let _ = compare(n_taps - 1, &taps.taps[n_taps - 1], &burn_audio);

    if let Some(i) = first_diverge {
        println!(
            "\n*** FIRST divergence (>0.1%): tap #{} {} {}",
            i, taps.taps[i].op_type, taps.taps[i].tensor
        );
    } else {
        println!("\nAll taps within 0.1% of ORT.");
    }

    // Dump full Burn values for the STFT/phase-chain taps to compare against
    // ORT (ort_full_taps.json) element-wise.
    let mut full = serde_json::Map::new();
    let phase_chain_idx = (
        12usize, // STFT
        173usize, 174usize, 178usize, 182usize, // Gather_4/5, Atan, Where_1
    );
    let names = [
        ("STFT_output_0", &burn_taps[phase_chain_idx.0]),
        ("Gather_4_output_0", &burn_taps[phase_chain_idx.1]),
        ("Gather_5_output_0", &burn_taps[phase_chain_idx.2]),
        ("Atan_output_0", &burn_taps[phase_chain_idx.3]),
        ("Where_1_output_0", &burn_taps[phase_chain_idx.4]),
    ];
    for (name, vals) in names.iter() {
        full.insert(name.to_string(), serde_json::json!(vals));
    }
    fs::write(
        artifacts.join("burn_full_taps.json"),
        serde_json::to_string(&full).unwrap(),
    )
    .expect("write burn_full_taps.json");
    println!("Wrote burn_full_taps.json");
}
