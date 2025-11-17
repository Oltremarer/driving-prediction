import json
import random

def generate_risk_list():
    """
    生成一个随机的、大致递增的风险值列表。
    """
    # 随机决定列表的长度，例如在 100 到 200 之间
    list_length = random.randint(100, 200)
    
    # 生成一堆随机浮点数
    # 为了保证列表的递增趋势，我们先生成随机数然后排序
    # -4 是为了给开头的 0.0 和结尾的 1.0 留出位置
    risk_values = [random.random() for _ in range(list_length - 4)]
    
    # 对随机数进行排序，以创建递增趋势
    risk_values.sort()

    # 模仿示例数据，在开头和结尾添加一些值
    # 示例中有时以 0.0 开头，以 1.0 结尾
    # 我们随机决定是否添加这些特殊值，使其更具多样性
    final_list = []
    if random.choice([True, False]):
        final_list.extend([0.0, 0.0]) # 随机加入0或2个0.0
        
    final_list.extend(risk_values)
    
    if random.choice([True, False]):
        final_list.append(1.0) # 随机加入1个1.0

    # 确保最终值至少有一个接近1.0
    if final_list[-1] < 0.98:
         final_list.append(random.uniform(0.98, 1.0))
         
    return final_list

def generate_data(num_records):
    """
    生成指定数量的视频数据记录。
    """
    all_data = {}
    print(f"正在生成 {num_records} 条数据...")

    for i in range(1, num_records + 1):
        # 1. 创建顶层键，如 "video_1", "video_2", ...
        video_key = f"video_{i}"
        
        # 2. 生成随机的 video_id，格式为 "数字/三位补零数字"
        # 例如 5/020, 11/092
        part1 = random.randint(1, 50)
        part2 = random.randint(0, 999)
        video_id_val = f"{part1}/{part2:03d}" # :03d 表示格式化为3位整数，不足补零
        
        # 3. 生成 risk 列表
        risk_list = generate_risk_list()
        
        # 4. 组装成最终的字典结构
        all_data[video_key] = {
            "video_id": video_id_val,
            "risk": risk_list
        }
        
        # 打印进度
        if i % 500 == 0:
            print(f"已生成 {i}/{num_records}...")
            
    print("数据生成完毕！")
    return all_data

if __name__ == "__main__":
    # 需要生成的记录数量
    NUM_RECORDS = 9334
    
    # 生成数据
    generated_data = generate_data(NUM_RECORDS)
    
    # 将数据写入 JSON 文件
    output_filename = "generated_data.json"
    print(f"正在将数据写入文件: {output_filename} ...")
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # indent=2 使 JSON 文件格式化，更易读
            json.dump(generated_data, f, indent=2)
        print("文件写入成功！")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")
