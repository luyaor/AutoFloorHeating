import matplotlib.pyplot as plt
import numpy as np

def plot_pipe_pt_seq(pipe_pt_seq):
    """
    根据 pipe_pt_seq 数据绘制管道布线路径

    参数:
        pipe_pt_seq: List[List[np.ndarray]]，每个内层列表代表一条路径，每个数组的 shape 为 (2,) 分别为 (x, y)
    """
    plt.figure(figsize=(10, 8))
    
    # 遍历所有路径
    for idx, route in enumerate(pipe_pt_seq):
        # 从每个 numpy 数组中提取 x 和 y 坐标
        xs = [pt[0] for pt in route]
        ys = [pt[1] for pt in route]
        
        # 绘制线条, 同时标记出路径点
        plt.plot(xs, ys, marker='o', label=f"Route {idx+1}")
    
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("管道布线路径示意图")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")  # 保持 x,y 轴比例一致
    plt.show()

if __name__ == "__main__":
    # 以下示例数据仅用于演示。你可以将此处换成实际的 pipe_pt_seq 数据
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
    
    plot_pipe_pt_seq(pipe_pt_seq) 