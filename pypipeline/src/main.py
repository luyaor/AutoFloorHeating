import partition
import cactus

def run_pipeline(case_id: int, num_x: int = 3, num_y: int = 4):
    """
    运行管道布线的完整流程
    
    Args:
        case_id: 测试用例编号 (0-5)
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    print(f"Processing case {case_id}...")
    
    # 1. 执行分区
    print("Performing partition...")
    partition.work(case_id, num_x=num_x, num_y=num_y)
    
    # 2. 执行管道布线
    print("Running pipe routing...")
    cactus.test_gen_all_color_m1()
    
    print("Pipeline completed!")

def main():
    # 先只测试case 4
    case_id = 4
    print(f"\n{'='*50}")
    print(f"Running case {case_id}")
    print('='*50)
    
    run_pipeline(case_id, num_x=1, num_y=2)

if __name__ == "__main__":
    main() 