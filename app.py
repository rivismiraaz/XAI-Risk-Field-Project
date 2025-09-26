import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import torch
import os
import json

from core.model import OrthogonalGDIFNet
from core.graph_constructor import create_2d_grid_graph
from simulators.modality_generator import ModalityGenerator

# --- 1. 全局初始化 ---
print("正在初始化Web应用...")
# 加载配置
with open('configs/default_config.json', 'r') as f:
    config = json.load(f)

DIMS, RISK_TYPES = config['grid']['dimensions'], config['risk_types']
W, H = DIMS
MODEL_CFG = config['model']

# 定义输入模态的映射关系和顺序
INPUT_MAP = {
    "collision": ["static_obstacles", "no_fly_zones"], "exposure": ["radar_coverage", "camera_coverage"],
    "energy": ["uav_state_battery"], "communication": ["comm_signal"], "threat": ["hostile_threat"]
}
DATA_CHANNELS = [
    "static_obstacles", "no_fly_zones", "radar_coverage", "camera_coverage",
    "comm_signal", "hostile_threat", "uav_state_battery"
]

# 初始化设备、图、模拟器
device = torch.device("cpu") # web应用通常在CPU上运行以获得更好的并发性
edge_index = create_2d_grid_graph(DIMS).to(device)
mod_gen = ModalityGenerator(DIMS)

# --- 2. 动态模型加载 ---
# 创建一个字典来缓存加载的模型，避免重复读取
loaded_models = {}
def get_model(scenario_name):
    if scenario_name not in loaded_models:
        print(f"正在为场景 '{scenario_name}' 加载模型...")
        model_path = os.path.join("models", f"{scenario_name}_model.pth")
        if not os.path.exists(model_path):
            print(f"错误: 未找到模型文件 {model_path}！")
            return None
        model = OrthogonalGDIFNet(RISK_TYPES, INPUT_MAP, **MODEL_CFG).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        loaded_models[scenario_name] = model
    return loaded_models[scenario_name]

# --- 3. Dash 应用布局 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("可信无人机集群的可解释风险场 (XAI Risk Field)", className="text-center text-primary, mb-3"))),
    dbc.Row([
        # 左侧控制面板
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("场景与可视化控制"),
                dbc.CardBody([
                    html.Label("选择作战场景 (加载对应模型)"),
                    dcc.Dropdown(id='scenario-selector', value='urban',
                                 options=[{'label': s.capitalize(), 'value': s} for s in ['urban', 'military', 'rural']]),
                    html.Hr(),
                    html.Label("选择可视化风险维度"),
                    dcc.Dropdown(id='risk-type-selector', value='collision',
                                 options=[{'label': r.capitalize(), 'value': r} for r in RISK_TYPES]),
                ])
            ]),
            dbc.Card(className="mt-3", children=[
                dbc.CardHeader("参数实时模拟"),
                dbc.CardBody([
                    html.Label("无人机当前电量"),
                    dcc.Slider(id='battery-slider', min=0, max=1, step=0.05, value=0.9, marks={0:'0%', 1:'100%'}, tooltip={"placement": "bottom"}),
                    html.Label("敌方威胁目标位置 (X, Y)"),
                    dcc.RangeSlider(id='hostile-pos-slider', min=0, max=W, step=1, value=[W/2, H/2], marks=None),
                ])
            ]),
            dbc.Card(id='inspector-card', className="mt-3", style={'minHeight': '280px'}),
        ], width=3),
        # 右侧图表
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id='2d-contour-map', style={'height': '85vh'}), label="2D 俯视等高线热图"),
                dbc.Tab(dcc.Graph(id='3d-surface-map', style={'height': '85vh'}), label="3D 等角风险地形图"),
            ])
        ], width=9),
    ]),
], fluid=True)

# --- 4. Dash 应用回调 (核心交互逻辑) ---
@app.callback(
    Output('2d-contour-map', 'figure'),
    Output('3d-surface-map', 'figure'),
    Output('inspector-card', 'children'),
    [Input('scenario-selector', 'value'), Input('risk-type-selector', 'value'),
     Input('battery-slider', 'value'), Input('hostile-pos-slider', 'value')],
    Input('2d-contour-map', 'hoverData')
)
def update_dashboard(scenario, risk_to_show, battery_level, hostile_pos, hover_data):
    # 1. 加载选定场景的模型
    model = get_model(scenario)
    if model is None: return go.Figure(), go.Figure(), "模型加载失败！"

    # 2. 根据UI输入生成模拟的模态数据
    sim_params = {
        'obstacles': [{'center': [20,20], 'scale_x': 10, 'scale_y': 10, 'amplitude': 1}, {'center': [70,80], 'scale_x': 15, 'scale_y': 10, 'amplitude': 0.8}],
        'nfz': [{'center': [80,20], 'scale_x': 20, 'scale_y': 20, 'amplitude': 1}],
        'radars': [{'center': [50,50], 'scale_x': 30, 'scale_y': 30, 'amplitude': 1}],
        'comm_stations': [{'center': [10,90], 'radius': 60}, {'center': [90,10], 'radius': 60}],
        'hostiles': [{'center': hostile_pos, 'scale_x': 8, 'scale_y': 8, 'amplitude': 1}],
        'battery_level': battery_level
    }
    modalities = mod_gen.generate_modalities(sim_params)
    
    # 3. 准备模型输入
    input_data_np = np.stack([modalities[name] if name in modalities else np.zeros(DIMS) for name in DATA_CHANNELS], axis=-1)
    data = torch.from_numpy(input_data_np.astype(np.float32)).reshape(-1, len(DATA_CHANNELS)).to(device)
    data_dict = {name: data[:, i].unsqueeze(1) for i, name in enumerate(DATA_CHANNELS)}

    # 4. 运行模型推理
    with torch.no_grad():
        risk_vector, _ = model(data_dict, edge_index)
    risk_vector_np = risk_vector.cpu().numpy().reshape(W, H, len(RISK_TYPES))

    # 5. 准备可视化
    risk_idx = RISK_TYPES.index(risk_to_show)
    data_to_plot = risk_vector_np[:, :, risk_idx]
    
    x_grid, y_grid = np.linspace(0, W, W), np.linspace(0, H, H)
    
    # 2D 图
    fig_2d = go.Figure(data=go.Contour(
        z=data_to_plot.T, x=x_grid, y=y_grid, colorscale='Reds', line_smoothing=0.85,
        colorbar_title=risk_to_show.capitalize(), contours=dict(coloring='heatmap')
    ))
    fig_2d.update_layout(title=f"2D 热图: {risk_to_show.capitalize()}", template="plotly_dark")

    # 3D 图
    fig_3d = go.Figure(data=go.Surface(z=data_to_plot.T, x=x_grid, y=y_grid, colorscale='Reds', cmin=0, cmax=1))
    fig_3d.update_layout(title=f"3D 地形图: {risk_to_show.capitalize()}", template="plotly_dark",
                        scene=dict(zaxis_title='Risk Level', aspectratio=dict(x=1, y=1, z=0.5)),
                        scene_camera_eye=dict(x=1.8, y=1.8, z=1.8))

    # 探针信息
    inspector_content = [dbc.CardHeader("悬停探针"), dbc.CardBody("将鼠标悬停在2D图上查看该点完整的风险向量")]
    if hover_data:
        point = hover_data['points'][0]
        x_idx, y_idx = int(point['x']), int(point['y'])
        if 0 <= x_idx < W and 0 <= y_idx < H:
            risk_values = risk_vector_np[y_idx, x_idx, :]
            inspector_content = [
                dbc.CardHeader(f"点 ({x_idx}, {y_idx}) 风险向量"),
                dbc.CardBody([html.P(f"{name.capitalize()}: {value:.3f}",
                                     style={'color':'red', 'font-weight':'bold'} if value > 0.7 else {}) 
                              for name, value in zip(RISK_TYPES, risk_values)])
            ]
            
    return fig_2d, fig_3d, inspector_content

# --- 5. 启动服务器 ---
if __name__ == '__main__':
    print("应用初始化完成。请在浏览器中打开 http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)
