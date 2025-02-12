import ezdxf
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os

def export_to_dxf(design_file: str, output_file: str = None) -> str:
    """
    将AR设计文件导出为DXF格式
    
    Args:
        design_file: AR设计JSON文件路径
        output_file: 输出DXF文件路径，如果不指定则使用相同文件名但扩展名为.dxf
        
    Returns:
        生成的DXF文件路径
    """
    # 1. 加载JSON数据
    with open(design_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 如果没有指定输出文件，则使用输入文件名但改为.dxf扩展名
    if output_file is None:
        output_file = str(Path(design_file).with_suffix('.dxf'))
    
    # 3. 创建新的DXF文档
    doc = ezdxf.new('R2010')  # 使用AutoCAD 2010格式
    msp = doc.modelspace()
    
    # 4. 为不同类型的实体创建图层
    doc.layers.new('WALLS', dxfattribs={'color': 1})  # 蓝色
    doc.layers.new('DOORS', dxfattribs={'color': 2})  # 黄色
    doc.layers.new('ROOMS', dxfattribs={'color': 3})  # 绿色
    doc.layers.new('TEXT', dxfattribs={'color': 7})   # 白色
    
    # 坐标缩放因子
    scale = 0.001  # 将毫米转换为米
    
    # 5. 遍历每个楼层
    for floor in data.get('Floor', []):
        if 'Construction' not in floor:
            continue
            
        construction = floor['Construction']
        
        # 绘制房间
        for room in construction.get('Room', []):
            # 绘制房间边界
            for boundary in room.get('Boundary', []):
                start = boundary.get('StartPoint', {})
                end = boundary.get('EndPoint', {})
                if start and end:
                    msp.add_line(
                        (start['x'] * scale, start['y'] * scale),
                        (end['x'] * scale, end['y'] * scale),
                        dxfattribs={'layer': 'ROOMS'}
                    )
            
            # 添加房间名称文本
            if 'AnnotationPoint' in room:  # 使用注释点作为文本位置
                point = room['AnnotationPoint']
                msp.add_text(
                    room.get('Name', ''),
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': 0.2,  # 文本高度也需要缩放
                        'insert': (point['x'] * scale, point['y'] * scale)
                    }
                )
        
        # 绘制门
        for door in construction.get('DoorAndWindow', []):
            if door.get('Type') == '门':  # 只处理门
                base_line = door.get('BaseLine', {})
                if base_line:
                    start = base_line.get('StartPoint', {})
                    end = base_line.get('EndPoint', {})
                    if start and end:
                        msp.add_line(
                            (start['x'] * scale, start['y'] * scale),
                            (end['x'] * scale, end['y'] * scale),
                            dxfattribs={'layer': 'DOORS'}
                        )
    
    # 6. 保存DXF文件
    doc.saveas(output_file)
    
    return output_file

def get_available_json_files():
    """获取data目录下所有可用的AR设计JSON文件"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("pypipeline/data")  # 尝试备选路径
    return sorted([f for f in data_dir.glob("ARDesign*.json")])

def main():
    """测试DXF导出功能"""
    print("\n=== DXF导出工具 ===")
    
    # 获取可用的JSON文件
    json_files = get_available_json_files()
    
    if not json_files:
        print("\n❌ 错误: 在data目录下未找到AR设计JSON文件")
        return
    
    print("\n🔷 可用的设计文件:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {file.name}")
    
    # 默认使用第一个文件
    selected_file = json_files[0]
    
    # 如果有多个文件，让用户选择
    if len(json_files) > 1:
        while True:
            choice = input(f"\n请选择要转换的文件 [1-{len(json_files)}，默认1]: ").strip()
            if not choice:  # 使用默认值
                break
            try:
                index = int(choice) - 1
                if 0 <= index < len(json_files):
                    selected_file = json_files[index]
                    break
                else:
                    print("❌ 无效的选择，请重试")
            except ValueError:
                print("❌ 请输入有效的数字")
    
    print(f"\n🔷 正在处理文件: {selected_file.name}")
    
    try:
        # 导出DXF文件
        output_file = export_to_dxf(str(selected_file))
        print(f"\n✅ DXF文件已成功导出至: {output_file}")
    except Exception as e:
        print(f"\n❌ 导出过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 