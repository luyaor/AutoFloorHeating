import json
import math
import numpy as np

def point_to_jpoint(pt):
    """
    将一个二维点（x,y，z默认为0）转换为 JPoint 格式
    """
    return {
        "x": float(pt[0]),
        "y": float(pt[1]),
        "z": 0.0
    }

def create_jline(pt1, pt2, curve_type=0):
    """
    根据两个二维点生成 JLine 格式数据

    参数:
        pt1, pt2: 数组或列表，代表两个点坐标
        curve_type: 曲线类型，0 表示直线
    """
    return {
        "StartPoint": point_to_jpoint(pt1),
        "EndPoint": point_to_jpoint(pt2),
        "CurveType": curve_type
    }

def compute_loop_length(route):
    """
    计算一条路径（回路）的总长度
    
    参数:
        route: 路径点序列，每个点的坐标单位为米
        
    返回:
        float: 路径总长度，单位为米
    """
    length = 0.0
    for i in range(len(route) - 1):
        x1, y1 = route[i]
        x2, y2 = route[i+1]
        # 计算两点之间的欧几里得距离
        distance = math.hypot(x2 - x1, y2 - y1)
        # 由于输入坐标已经是米为单位，所以这里不需要额外转换
        length += distance
    return round(length, 2)  # 保留2位小数

def convert_route_to_path(route):
    """
    将一条路径（回路）中的连续点对转换为 JLine 列表

    参数:
        route: List[np.ndarray] 每个点的坐标单位为米
    返回:
        List[JLine]，每个 JLine 为字典形式，坐标单位为毫米
    """
    lines = []
    for i in range(len(route) - 1):
        # 转换为毫米单位，因为输出要求是毫米
        pt1 = route[i] * 1000  # 米转毫米
        pt2 = route[i+1] * 1000  # 米转毫米
        line = create_jline(pt1, pt2, curve_type=0)
        lines.append(line)
    return lines

def convert_pipe_pt_seq_to_heating_design(pipe_pt_seq, 
                                          level_name="1F", 
                                          level_no=1, 
                                          level_desc="首层", 
                                          house_name="house_XYZ", 
                                          curvity=100,
                                          input_data=None,
                                          input_scale=0.01):  # 新增参数，用于将输入单位转换为米, 默认假设输入单位为厘米
    """
    将 pipe_pt_seq 数据转换为符合输出要求的地暖设计数据。
    
    参数:
        pipe_pt_seq: 管道点序列，每个点的单位依据 input_scale 转换到米
        input_scale: 输入数据的缩放比例，将输入坐标转换为米。默认值0.01（即输入单位为厘米）
        其它参数同前
    """
    coil_loops = []
    for route in pipe_pt_seq:
        # 将每个点转换为米
        route_m = [np.array([pt[0] * input_scale, pt[1] * input_scale]) for pt in route]
        loop_length = compute_loop_length(route_m)
        jline_path = convert_route_to_path(route_m)
        coil_loop = {
            "Length": loop_length,
            "Areas": [],       # 可从 input_data 中进一步填充
            "Path": jline_path,
            "Curvity": curvity
        }
        coil_loops.append(coil_loop)
    
    collector_coil = {
        "CollectorName": "Collector_1",
        "Loops": len(pipe_pt_seq),
        "CoilLoops": coil_loops,
        "Deliverys": []  # 如果 input_data 中有输配管信息，可赋值
    }
    
    # 从 input_data 中获取伸缩缝（Expansions）的数据（字段名根据实际情况确定）
    expansions = []
    if input_data is not None:
        # 假设 input_data 中有一个键 "ExpansionsData" 存放伸缩缝信息
        expansions = input_data.get("ExpansionsData", [])
    
    heating_coil = {
        "LevelName": level_name,
        "LevelNo": level_no,
        "LevelDesc": level_desc,
        "HouseName": house_name,
        "Expansions": expansions,  # 这里就不再为空列表
        "CollectorCoils": [collector_coil]
    }
    
    # 直接返回 heating_coil 对象，不需要额外的 Heating 包装
    return heating_coil

def save_design_to_json(design_data, out_file):
    """
    将设计数据写入 JSON 文件
    """
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(design_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 这里给出一个示例 pipe_pt_seq 数据，实际请替换为 solve_pipeline 得到的数据
    pipe_pt_seq = [
        [
            np.array([135.12490417, 93.24979167]),
            np.array([135.12490417, 71.97049052]),
            np.array([136.48519526, 69.24990833]),
            np.array([137.39642649, 69.24990833]),
            np.array([138.45834167, 68.18799316]),
            np.array([138.45834167, 64.40652616]),
            np.array([138.5      , 63.4481845]),
            np.array([138.5      , 58.98778771]),
            np.array([138.65625  , 55.73778771]),
            np.array([138.65625  , 4.20564466]),
            np.array([143.16541801, 1.84365247]),
            np.array([147.8649474 , 1.84364902]),
            np.array([148.86495387, 2.15614204]),
            np.array([153.63514709, 2.15613855]),
            np.array([154.63515311, 1.84364406]),
            np.array([168.40348744, 1.84363398]),
            np.array([169.99724435, 2.74987656]),
            np.array([270.01776814, 2.74980328]),
            np.array([273.25     , 5.98203278]),
            np.array([273.25     , 53.01756695]),
            np.array([270.01776695, 56.2498    ]),
            np.array([209.23218305, 56.2498    ]),
            np.array([205.99995   , 53.01756695]),
            np.array([205.99995   , 11.29700398]),
            np.array([207.18233518, 7.74984932])
        ],
        [
            np.array([136.1249375 , 93.24979167]),
            np.array([136.1249375 , 71.97049052]),
            np.array([136.48519526, 71.249975  ]),
            np.array([137.42377104, 71.249975  ]),
            np.array([142.125075  , 68.09005185]),
            np.array([142.125075  , 64.36724873]),
            np.array([142.5       , 62.64223373]),
            np.array([142.5       , 58.90605   ]),
            np.array([143.875     , 57.8748   ]),
            np.array([146.06372965, 57.8748    ]),
            np.array([148.56372965, 58.7498    ]),
            np.array([153.92583536, 58.7498    ]),
            np.array([156.42583536, 58.2498    ]),
            np.array([164.20783797, 58.2498    ]),
            np.array([167.7501    , 60.37515722]),
            np.array([167.7501    , 88.07577486]),
            np.array([172.69991787, 94.53211518]),
            np.array([177.48223305, 89.7498    ])
        ]
    ]
    
    design_data = convert_pipe_pt_seq_to_heating_design(pipe_pt_seq, 
                                                        level_name="1F",
                                                        level_no=1,
                                                        level_desc="首层",
                                                        house_name="c1c37dc1a40f4302b6552a23cd1fd557",
                                                        curvity=100)
    out_file = "HeatingDesign_output.json"
    save_design_to_json(design_data, out_file)
    print(f"转换后的地暖设计数据已保存到：{out_file}") 