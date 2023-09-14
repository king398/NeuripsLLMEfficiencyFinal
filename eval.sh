python main.py \
  --model hf-causal-experimental \
  --model_args pretrained=meta-llama/Llama-2-13b-hf,dtype="float16",peft=/home/mithil/PycharmProjects/NeuripsLLMEfficiency/models/Llama-2-13b-hf-lr-1e-4-all-module/checkpoint-10035/,use_accelerate=True \
  --tasks hendrycksTest-philosophy,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,bigbench_causal_judgement,bigbench_logical_deduction_three_objects,bigbench_snarks,truthfulqa_mc,gsm8k \
  --output_path results/Llama-2-7b-baseline-bigbench.json \
  --batch_size 4
