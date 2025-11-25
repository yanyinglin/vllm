from transformers import AutoModelForCausalLM

# 加载单个stage
stage0 = AutoModelForCausalLM.from_pretrained("/home/yanying/pipeline_export/Llama-3-8B/stage_0", device_map="cuda:0")
stage1 = AutoModelForCausalLM.from_pretrained("/home/yanying/pipeline_export/Llama-3-8B/stage_1", device_map="cuda:1")

print("=== Stage 0 Model Info ===")
print(f"Model: {stage0}")
print(f"Model dtype: {stage0.dtype}")
print(f"Model device: {stage0.device}")
if hasattr(stage0, 'config'):
    print(f"Model config: {stage0.config}")

# Print parameter dtypes for stage0
print("\nStage 0 Parameter dtypes:")
for name, param in stage0.named_parameters():
    print(f"  {name}: {param.dtype}")
    break  # Just show first parameter as example

print("\n=== Stage 1 Model Info ===")
print(f"Model: {stage1}")
print(f"Model dtype: {stage1.dtype}")
print(f"Model device: {stage1.device}")
if hasattr(stage1, 'config'):
    print(f"Model config: {stage1.config}")

# Print parameter dtypes for stage1
print("\nStage 1 Parameter dtypes:")
for name, param in stage1.named_parameters():
    print(f"  {name}: {param.dtype}")
    break  # Just show first parameter as example