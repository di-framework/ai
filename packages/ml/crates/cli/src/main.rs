use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use di_ml::{
    default_init_dir, exists_model, run_workspace, scaffold, Error, InitSpec, LossKind,
    OptimizerKind, Result, TrainConfig, DEFAULT_INIT_DIR,
};
use inquire::{Confirm, CustomType, InquireError, Select, Text};

#[derive(Parser)]
#[command(name = "di-ml")]
#[command(version)]
#[command(about = "Fine-tune an ONNX model from a directory workspace")]
#[command(args_conflicts_with_subcommands = true)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Workspace directory containing model.onnx, train.toml, and data/
    workspace: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a training workspace (XOR demo)
    Init {
        /// Directory to create (default: ./xor-workspace)
        dir: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(err.exit_code() as u8)
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Init { dir }) => cmd_init(dir),
        None => {
            let Some(workspace) = cli.workspace else {
                return Err(Error::usage(
                    "missing workspace directory; pass a path or run `di-ml init`",
                ));
            };
            cmd_train(workspace)
        }
    }
}

fn cmd_train(workspace: PathBuf) -> Result<()> {
    let result = run_workspace(&workspace)?;
    println!("accel {}", result.report.accel);
    if result.report.peak_bytes > 0 {
        println!(
            "peak-bytes {}  step-ms {:.2}",
            result.report.peak_bytes, result.report.step_time_ms
        );
    }
    println!("trainable {}", result.report.trainable.len());
    for row in &result.report.history {
        match row.eval_loss {
            Some(eval) => println!(
                "step {:>5}  train={:.4}  eval={:.4}",
                row.step, row.train_loss, eval
            ),
            None => println!("step {:>5}  train={:.4}", row.step, row.train_loss),
        }
    }
    println!("wrote {}", result.dist_model.display());
    println!("wrote {}", result.dist_metrics.display());
    if let Some(eval) = result.report.final_eval_loss {
        println!(
            "final train loss {:.4}  eval loss {:.4}",
            result.report.final_train_loss, eval
        );
    } else {
        println!("final train loss {:.4}", result.report.final_train_loss);
    }
    Ok(())
}

fn cmd_init(dir: Option<PathBuf>) -> Result<()> {
    let spec = if std::io::stdin().is_terminal() {
        prompt_init(dir)?
    } else {
        InitSpec::xor_demo(dir.unwrap_or_else(default_init_dir))
    };
    let root = scaffold(&spec)?;
    println!("wrote {}", root.join(di_ml::MODEL_FILE).display());
    println!("wrote {}", root.join(di_ml::TRAIN_TOML).display());
    println!(
        "wrote {}",
        root.join(di_ml::DATA_DIR).join(di_ml::TRAIN_JSONL).display()
    );
    println!("next: di-ml {}", root.display());
    Ok(())
}

fn prompt_init(dir: Option<PathBuf>) -> Result<InitSpec> {
    let root = match dir {
        Some(path) => path,
        None => {
            let value = Text::new("Workspace directory")
                .with_default(DEFAULT_INIT_DIR)
                .prompt()
                .map_err(inquire_err)?;
            PathBuf::from(value)
        }
    };

    let overwrite = if exists_model(&root) {
        Confirm::new(&format!(
            "{} already has model.onnx. Overwrite the workspace files?",
            root.display()
        ))
        .with_default(false)
        .prompt()
        .map_err(inquire_err)?
    } else {
        false
    };

    let loss = select(
        "Loss",
        &["mse", "cross-entropy", "bce-logits", "l1", "infonce", "cosine"],
        "mse",
    )?;
    let optimizer = select("Optimizer", &["adamw", "sgd"], "adamw")?;
    let learning_rate = CustomType::<f32>::new("Learning rate")
        .with_default(0.05)
        .prompt()
        .map_err(inquire_err)?;
    let max_steps = CustomType::<usize>::new("Max steps")
        .with_default(400)
        .prompt()
        .map_err(inquire_err)?;
    let batch_size = CustomType::<usize>::new("Batch size")
        .with_default(4)
        .prompt()
        .map_err(inquire_err)?;
    let seed = CustomType::<u64>::new("Seed")
        .with_default(1)
        .prompt()
        .map_err(inquire_err)?;

    Ok(InitSpec {
        root,
        config: TrainConfig {
            loss: LossKind::parse(&loss)?,
            optimizer: OptimizerKind::parse(&optimizer)?,
            learning_rate,
            max_steps,
            batch_size,
            seed,
            log_every: 50,
            ..TrainConfig::default()
        },
        overwrite,
    })
}

fn select(label: &str, options: &'static [&'static str], default: &str) -> Result<String> {
    let start = options.iter().position(|o| *o == default).unwrap_or(0);
    Select::new(label, options.to_vec())
        .with_starting_cursor(start)
        .prompt()
        .map(|s| s.to_string())
        .map_err(inquire_err)
}

fn inquire_err(err: InquireError) -> Error {
    match err {
        InquireError::OperationCanceled | InquireError::OperationInterrupted => {
            Error::usage("init cancelled")
        }
        other => Error::fail(other.to_string()),
    }
}
