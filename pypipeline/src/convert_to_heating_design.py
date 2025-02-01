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
    计算一条路径（回路）的总长度，路径为 numpy 数组列表
    """
    length = 0.0
    for i in range(len(route) - 1):
        x1, y1 = route[i]
        x2, y2 = route[i+1]
        distance = math.hypot(x2 - x1, y2 - y1)
        length += distance
    return length

def convert_route_to_path(route):
    """
    将一条路径（回路）中的连续点对转换为 JLine 列表

    参数:
        route: List[np.ndarray] 每个 numpy 数组为 (2,)
    返回:
        List[JLine]，每个 JLine 为字典形式
    """
    lines = []
    for i in range(len(route) - 1):
        line = create_jline(route[i], route[i+1], curve_type=0)
        lines.append(line)
    return lines

def convert_pipe_pt_seq_to_heating_design(pipe_pt_seq, 
                                          level_name="1F", 
                                          level_no=1, 
                                          level_desc="首层", 
                                          house_name="house_XYZ", 
                                          curvity=100):
    """
    将 pipe_pt_seq（List[List[np.ndarray]]）转换为符合文档要求的地暖设计输出格式：
    
    输出格式：
    {
      "Heating": {
          "HeatingCoil": [
              {
                  "LevelName": <string>,
                  "LevelNo": <int>,
                  "LevelDesc": <string>,
                  "HouseName": <string>,
                  "Expansions": [List<JLine>],
                  "CollectorCoils": [
                      {
                          "CollectorName": <string>,
                          "Loops": <int>,
                          "CoilLoops": [
                              {
                                  "Length": <float>,
                                  "Areas": [List],
                                  "Path": [List<JLine>],
                                  "Curvity": <int>
                              },
                              ... (每个回路)
                          ],
                          "Deliverys": [List<List<JLine>>]
                      }
                  ]
              }
          ],
          "Risers": []
      }
    }
    """
    coil_loops = []
    # 对于每条路径，计算长度并转换为 Path（JLine 列表）
    for route in pipe_pt_seq:
        loop_length = compute_loop_length(route)
        jline_path = convert_route_to_path(route)
        coil_loop = {
            "Length": loop_length,
            "Areas": [],       # 如果需要，可以按照其它逻辑计算区域，暂置空列表
            "Path": jline_path,
            "Curvity": curvity
        }
        coil_loops.append(coil_loop)
    
    collector_coil = {
        "CollectorName": "Collector_1",
        "Loops": len(pipe_pt_seq),
        "CoilLoops": coil_loops,
        "Deliverys": []  # 暂置为空列表
    }
    
    heating_coil = {
        "LevelName": level_name,
        "LevelNo": level_no,
        "LevelDesc": level_desc,
        "HouseName": house_name,
        "Expansions": [],  # 如果有伸缩缝数据，可以填入
        "CollectorCoils": [collector_coil]
    }
    
    design_data = {
        "Heating": {
            "HeatingCoil": [heating_coil],
            "Risers": []
        }
    }
    
    return design_data

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