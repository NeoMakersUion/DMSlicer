import streamlit as st
import time

# Set page configuration
st.set_page_config(
    page_title="Geom Practice Tests",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define content for each page
def home_page():
    st.title("欢迎使用几何练习测试应用")
    st.markdown("""
    这是一个基于 Streamlit 构建的练习测试平台。
    
    ### 功能特点：
    - **侧边栏导航**：轻松切换不同的练习测试。
    - **实时交互**：即时反馈测试结果。
    - **独立内容**：每个页面专注于特定的测试主题。
    
    请在左侧侧边栏选择一个测试开始练习。
    """)
    
    st.info("👈 点击左侧侧边栏打开菜单")
    
    # Simple interactive element
    if st.button("点击这里查看示例交互"):
        st.balloons()
        st.success("交互成功！祝你练习愉快！")

def practice_test_1():
    st.title("练习测试 1: 基础几何")
    st.markdown("### 本测试涵盖点、线、面的基础知识。")
    
    with st.form("test_1_form"):
        st.subheader("问题 1: 三角形的内角和是多少度？")
        q1 = st.radio("选择答案：", ["90度", "180度", "270度", "360度"], key="q1")
        
        st.subheader("问题 2: 两点之间最短的距离是？")
        q2 = st.selectbox("选择答案：", ["直线", "折线", "曲线", "圆弧"], key="q2")
        
        submitted = st.form_submit_button("提交答案")
        if submitted:
            score = 0
            if q1 == "180度":
                score += 1
            if q2 == "直线":
                score += 1
            
            if score == 2:
                st.success(f"恭喜！你答对了所有问题！得分：{score}/2")
            else:
                st.warning(f"不错，继续努力！得分：{score}/2")

def practice_test_2():
    st.title("练习测试 2: 立体几何")
    st.markdown("### 本测试涵盖体积和表面积计算。")
    
    st.write("计算以下图形的体积：")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://placeholder.pics/svg/300x200/DEDEDE/555555/Cube", caption="正方体 (边长 = 2)")
        ans1 = st.number_input("正方体的体积是？", min_value=0.0, step=0.1)
    
    with col2:
        st.image("https://placeholder.pics/svg/300x200/DEDEDE/555555/Sphere", caption="球体 (半径 = 1)")
        ans2 = st.number_input("球体的体积是？(保留两位小数)", min_value=0.0, step=0.1)
    
    if st.button("检查答案"):
        with st.spinner("正在计算..."):
            time.sleep(0.5)  # Simulate processing time
        
        correct_cube = abs(ans1 - 8.0) < 0.01
        correct_sphere = abs(ans2 - 4.19) < 0.1 # 4/3 * pi * 1^3 approx 4.188...
        
        if correct_cube and correct_sphere:
            st.success("太棒了！全部正确！")
        elif correct_cube or correct_sphere:
            st.info("部分正确，请检查错误的题目。")
        else:
            st.error("答案不正确，请再试一次。")

def practice_test_3():
    st.title("练习测试 3: 解析几何")
    st.markdown("### 本测试涵盖坐标系和方程。")
    
    st.slider("选择直线的斜率", min_value=-10, max_value=10, value=1)
    st.text_input("请输入圆的标准方程：")
    
    st.write("（此页面演示更多交互控件，暂无评分逻辑）")
    
    data = {"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]}
    st.line_chart(data)

# Main App Structure
def main():
    # Sidebar Navigation
    st.sidebar.title("导航")
    
    # Define pages
    pages = {
        "主页": home_page,
        "练习测试 1 (基础)": practice_test_1,
        "练习测试 2 (立体)": practice_test_2,
        "练习测试 3 (解析)": practice_test_3
    }
    
    selection = st.sidebar.radio("前往：", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("选择上方的一个选项以切换页面。")
    
    # Display the selected page
    page_function = pages[selection]
    page_function()

if __name__ == "__main__":
    main()
