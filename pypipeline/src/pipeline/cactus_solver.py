import json
from pathlib import Path
from core import cactus
from core.cactus import CacRegion, CactusSolverDebug, arr
import os
import traceback

def load_solver_params(json_file):
    """从JSON文件加载求解器参数"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def solve_pipeline(intermediate_data_file: str):
    """
    执行管道布线计算
    
    Args:
        intermediate_data_file: 中间数据文件路径
    """
    print("🔷 正在加载布线模型...")
    
    # 加载中间数据
    loaded_params = load_solver_params(intermediate_data_file)
    print("✅ 已加载中间数据")
    
    seg_pts = loaded_params['seg_pts']
    regions = loaded_params['regions']
    wall_path = loaded_params['wall_path']
    destination_pt=loaded_params['destination_pt']
    suggested_m0_pipe_interval=loaded_params['pipe_interval']

    cmap_0 = [
        "blue",
        "yellow",
        "red",
        "cyan",
        "green",
        "purple",
        "orange",
        "pink",
        "brown",
    ]
    cmap = {
        -1: "black",
        0: "grey",
        **{(i + 1): cmap_0[i % len(cmap_0)] for i in range(100)},
    }

    print("🔷 开始计算管道布线方案...")
    print("seg_pts=", seg_pts)

    solver = cactus.CactusSolver(
        # glb_h=30000, 
        # glb_w=30000, 
        cmap=cmap, 
        # seg_pts=[arr(x[0] / 100 - 130, x[1] / 100) for x in seg_pts], 
        # seg_pts=seg_pts, 
        seg_pts=[arr(x[0], x[1]) for x in seg_pts],
        wall_pt_path=wall_path, 
        cac_regions=[CacRegion(x[0][::1], x[1]) for x in regions], 
        destination_pt=destination_pt, 
        suggested_m0_pipe_interval=suggested_m0_pipe_interval
    )
    
    pipe_pt_seq = solver.process(CactusSolverDebug(m1=False))
    print("✅ 管道布线计算完成!")
    
    return pipe_pt_seq

if __name__ == "__main__":
    print("\n🔷 Starting case data conversion...")
    from tools.case_converter import convert_all_cases
    convert_all_cases()
    print("\n✅ All cases converted successfully!") 
    cases_dir = "output"
    results = []
    
    # 遍历所有案例文件
    for file in sorted(os.listdir(cases_dir)):
        if file.endswith("_intermediate.json"):
            case_path = os.path.join(cases_dir, file)
            print(f"\n🔷 测试案例: {file}")
            try:
                solver = solve_pipeline(case_path)
                results.append((file, "成功"))
                print(f"✅ {file} 测试通过")
            except Exception as e:
                results.append((file, f"失败: {str(e)}"))
                print(f"❌ {file} 测试失败")
                print(f"错误信息: {str(e)}")
                print("详细错误:")
                traceback.print_exc()
    
    # 打印总结报告
    print("\n📊 测试结果总结:")
    for file, status in results:
        status_symbol = "✅" if status == "成功" else "❌"
        print(f"{status_symbol} {file}: {status}")