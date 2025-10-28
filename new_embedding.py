import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

from stress_test import *

def now() -> float:
    return time.perf_counter()

def maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def calculate_percentiles(arr: List[float], percentiles: List[int] = [90, 95, 99]) -> Dict[str, float]:
    if not arr: return {f"p{p}": float("nan") for p in percentiles}
    data = np.asarray(arr, dtype=float)
    percentile_values = np.percentile(data, percentiles)
    return {f"p{p}": val for p, val in zip(percentiles, percentile_values)}

class Embedding_e5():
    def __init__(self, model_path=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'intfloat/e5-mistral-7b-instruct'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 使用AutoModel加载模型本体
        # torch_dtype=torch.bfloat16 是在现代GPU上获得高性能的关键
        self.model = AutoModel.from_pretrained(
            "/home/judy/wjj/multi-model-rag/e5-mistral-7b-instruct",
            torch_dtype=torch.bfloat16,
            device_map=self.device  # 自动将模型加载到指定设备
        )
        self.model.eval() # 将模型设置为评估模式
        self.mode = 'e5-mistral'
        self.batch_sizes=[500,1000,1500,1600,1700,1800,2000,2100,2200]
        self.warmup = 2
        self.repeats=10

    def last_token_pool(self,last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def expand_batch(self,batchsize):
        input_texts = [
            'Instruct: How to bake a delicious chocolate cake?',
            'Passage: Baking a chocolate cake involves mixing flour, sugar, cocoa powder, and eggs. First, preheat your oven to 350°F (175°C). Then, grease and flour a 9-inch round baking pan.',
            'Passage: The history of the Eiffel Tower dates back to the 1889 Exposition Universelle (World\'s Fair) held in Paris.',
            'This is a standalone sentence without any prefix.'
        ]   
        # 扩充
        expanded_texts = [input_texts[i % len(input_texts)] for i in range(batchsize)]
        return expanded_texts

    def batch_encode(self,input_texts):
        with torch.no_grad():
            # 将文本列表分词，并移动到GPU
            batch_dict = self.tokenizer(
                input_texts, 
                max_length=4096//16,  # e5-mistral支持长达4096的上下文
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)   
        return embeddings

    def measure_embedding(self):
        for bsz in self.batch_sizes:
            batch_input = self.expand_batch(bsz)
            
            totals = []
            # 预热
            for _ in range(self.warmup):
                _=self.batch_encode(batch_input)

            for _ in range(self.repeats):
                
                maybe_cuda_sync()
                t0 = now()
                _=self.batch_encode(batch_input)
                maybe_cuda_sync()
                totals.append(now() - t0)

            percentile_results = calculate_percentiles(totals, [50,90, 95, 99])
            rec = {
                "mode": self.mode,
                "batch_size": bsz,
                "avg_total_s": float(np.mean(totals)),
                "avg_per_sample_s": float(np.mean(totals)) / bsz,
            }
            rec.update({f"{k}_total_s": v for k, v in percentile_results.items()})

            p_str = " | ".join([f"p{p}={v:.5f}s" for p, v in percentile_results.items()])
            print(
                f"[{self.mode:<9} bsz={bsz:<5d}] "
                f"Avg Total: {rec['avg_total_s']:.5f}s | "
                f"Avg Per-Sample: {rec['avg_per_sample_s']:.8f}s | "
                f"Percentiles: [ {p_str} ]"
            )
# model = Embedding_e5()  
# model.measure_embedding()
# HuggingFaceM4/idefics2-8b (多模态图文Embedding)，查考上面的实现，实现HuggingFaceM4/idefics2-8b embedding
# 图片path/home/judy/wjj/multi-model-rag/data/space.png
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForVision2Seq
class Embedding_idefics2():
    """
    使用 HuggingFaceM4/idefics2-8b 模型为图片生成 Embedding 的类。
    """
    def __init__(self, model_path='/home/judy/idefics2-8b', image_path='/home/judy/wjj/multi-model-rag/data/space.png'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_path = image_path
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            do_image_splitting=False 
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        self.mode = 'idefics2-8b-image'
        self.batch_sizes = [160,170,200,220,240]
        self.warmup = 2
        self.repeats = 10

    def expand_batch(self, batchsize: int) -> Dict[str, torch.Tensor]:
        # <-- 更改点 1: 此方法现在返回预处理后的张量字典，而不是图像列表。
        """
        加载、复制并预处理一批图像，返回模型所需的张量字典。
        """
        # 1. 加载并复制图像
        try:
            image = Image.open(self.image_path).convert("RGB")
        except FileNotFoundError:
            print(f"错误: 无法在路径 '{self.image_path}' 找到图片。请检查路径是否正确。")
            image = Image.new('RGB', (224, 224), color = 'red')
            print("已创建一个红色占位符图像继续运行。")

        images = [image] * batchsize

        # 2. <-- 更改点 2: 将 processor 的处理逻辑移到这里
        # 准备输入:
        # processor 会将图像转换为 pixel_values 张量，并为文本生成 input_ids。
        texts = ["<image>"] * batchsize # 空文本提示
        inputs = self.processor(
            text=texts, 
            images=images, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        return inputs

    def batch_encode(self, inputs: Dict[str, torch.Tensor]) -> Tensor:
        # <-- 更改点 3: 方法签名已更改，直接接收张量字典
        """
        对一批预处理好的数据进行编码，提取并返回它们的 embedding。
        """
        with torch.no_grad():
            # processor 的处理步骤已移出，这里直接进行模型推理
            outputs = self.model(**inputs, output_hidden_states=True)

            # 提取图像 Embedding 的逻辑保持不变
            hidden_states = outputs.hidden_states[-1]
            num_image_tokens = inputs['pixel_values'].shape[1]
            image_embeddings = hidden_states[:, 1:1 + num_image_tokens, :].mean(dim=1)
            
            # 归一化
            normalized_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            embeddings=normalized_embeddings.detach().cpu().float().numpy().astype(np.float32, copy=False)

        return embeddings

    def process_database(self):
        IMAGE_FOLDER = "/home/judy/wjj/cocodataset/formatted"  # coco下载数据集的位置
        OUTPUT_FOLDER = "/home/judy/wjj/cocodataset/vector4096"  # 输出索引和元数据的文件夹

        BATCH_SIZE = 32                        # 批处理大小，根据您的 GPU 显存调整

        # --- 1. 初始化模型和设备 ---
        print(">>> 步骤 1: 初始化模型和设备...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("警告：未检测到 CUDA，将使用 CPU。这会非常慢。")



        print("\n>>> 步骤 2: 生成图片 Embeddings...")
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        image_paths = list(glob.glob(f"{IMAGE_FOLDER}/**/*.jpg", recursive=True)) + \
            list(glob.glob(f"{IMAGE_FOLDER}/**/*.png", recursive=True))

        all_embeddings = []
        all_metadata = []


        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="处理图片批次"):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_images = []

            # 预处理图片
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"警告：无法处理图片 {path}，已跳过。错误: {e}")
                    continue

            if not batch_images:
                continue
            texts = ["<image>"] * len(batch_images) # 空文本提示
            inputs = self.processor(
                text=texts, 
                images=batch_images, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            


            image_features = self.batch_encode(inputs)

            # 将向量移回 CPU 并添加到列表中
            all_embeddings.append(image_features)
            all_metadata.extend(batch_paths)  # 记录对应的文件路径

        # 将所有批次的向量合并成一个大的 numpy 数组
        all_embeddings = np.vstack(all_embeddings)
        np.save(OUTPUT_FOLDER+"/all_embeddings.npy", all_embeddings)

        metadata_df = pd.DataFrame({'filepath': all_metadata})
        metadata_output_path = os.path.join(
            OUTPUT_FOLDER, "all_metadata.feather")  # [修正] 文件扩展名
        metadata_df.to_feather(metadata_output_path)
        print("embedding finished ~ ~")

    def measure_embedding(self):
        """
        测量不同批处理大小下的 embedding 生成性能。
        此方法无需更改，因为它只是将 expand_batch 的输出传递给 batch_encode。
        """
        print("\n--- Measuring Image Embedding (idefics2-8b) Performance ---")
        for bsz in self.batch_sizes:
            if bsz == 0: continue
            
            try:
                # `batch_input` 现在是一个包含张量的字典
                batch_input = self.expand_batch(bsz)
                
                totals = []
                # 预热
                for _ in range(self.warmup):
                    _ = self.batch_encode(batch_input)

                # 重复测量
                for _ in range(self.repeats):
                    maybe_cuda_sync()
                    t0 = now()
                    embeddings = self.batch_encode(batch_input)
                    maybe_cuda_sync()
                    totals.append(now() - t0)

                percentile_results = calculate_percentiles(totals, [50, 90, 95, 99])
                rec = {
                    "mode": self.mode,
                    "batch_size": bsz,
                    "avg_total_s": float(np.mean(totals)),
                    "avg_per_sample_s": float(np.mean(totals)) / bsz,
                    "embedding_dim": embeddings.shape[-1]
                }
                rec.update({f"{k}_total_s": v for k, v in percentile_results.items()})

                p_str = " | ".join([f"p{p}={v:.5f}s" for p, v in percentile_results.items()])
                print(
                    f"[{self.mode:<20} bsz={bsz:<5d}] "
                    f"Avg Total: {rec['avg_total_s']:.5f}s | "
                    f"Avg Per-Sample: {rec['avg_per_sample_s']:.8f}s | "
                    f"Dim: {rec['embedding_dim']:<4d} | "
                    f"Percentiles: [ {p_str} ]"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[{self.mode:<20} bsz={bsz:<5d}] CUDA out of memory. Skipping this batch size and larger.")
                break
            except Exception as e:
                print(f"[{self.mode:<20} bsz={bsz:<5d}] An error occurred: {e}")
                break

        
if __name__ == '__main__':
    # 运行 E5 文本 embedding 测试 (可选)
    # print("Initializing e5-mistral-7b-instruct model...")
    # text_model = Embedding_e5(model_path="/home/judy/wjj/multi-model-rag/e5-mistral-7b-instruct")  
    # text_model.measure_embedding()

    # 运行 Idefics2 图片 embedding 测试
    print("\nInitializing HuggingFaceM4/idefics2-8b model...")
    # 请确保这里的 image_path 是正确的
    image_model = Embedding_idefics2(
    )
    # image_model.measure_embedding()
    image_model.process_database()