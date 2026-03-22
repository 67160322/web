import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Video Game Sales Predictor", layout="wide")

# --- 2. โหลด Assets ---
@st.cache_resource
def load_all_assets():
    try:
        # 1. โหลดโมเดล
        if not os.path.exists('vgsales_model.pkl'):
            return None, None, None
        model = joblib.load('vgsales_model.pkl')
        
        # 2. โหลด Metadata (ใช้ชื่อไฟล์แบบไม่มี .json ตามที่คุณมีในเครื่อง)
        metadata_path = 'model_metadata'
        if not os.path.exists(metadata_path):
            metadata_path = 'model_metadata.json' # สำรองไว้เผื่อมีนามสกุล
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 3. โหลด Platform Mapping
        mapping_path = 'platform_mapping'
        if not os.path.exists(mapping_path):
            mapping_path = 'platform_mapping.json' # สำรองไว้เผื่อมีนามสกุล
            
        with open(mapping_path, 'r', encoding='utf-8') as f:
            platform_mapping = json.load(f)
            
        return model, metadata, platform_mapping
    except Exception as e:
        # ส่งค่า Error กลับไปแสดงผล
        return None, None, str(e)

model, metadata, platform_mapping = load_all_assets()

# ตรวจสอบการโหลดข้อมูล
if model is None:
    st.error(f"❌ โหลดข้อมูลไม่สำเร็จ: {platform_mapping}")
    st.info("คำแนะนำ: ตรวจสอบว่าไฟล์ vgsales_model.pkl, model_metadata และ platform_mapping อยู่ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# --- 3. ส่วน UI ---
st.title("🎮 Video Game Sales Prediction App")
st.markdown("แอปพลิเคชันทำนายยอดขายเกมทั่วโลกด้วย Machine Learning")
st.markdown("---")

# --- 4. Sidebar สำหรับ Prediction ---
st.sidebar.header("🕹️ ระบุข้อมูลเกม")

# ดึงข้อมูลจาก metadata
genres = metadata.get('genres', [])
genre = st.sidebar.selectbox("แนวเกม (Genre)", genres)

# Filter Platform ตาม Genre
available_platforms = platform_mapping.get(genre, metadata.get('platforms', []))
platform = st.sidebar.selectbox("เครื่องเล่น (Platform)", available_platforms)

publishers = metadata.get('publishers', [])
publisher = st.sidebar.selectbox("ผู้จัดจำหน่าย (Publisher)", publishers)
year = st.sidebar.number_input("ปีที่วางขาย (Year)", 1980, 2025, 2024)

# ส่วนการทำนาย
if st.sidebar.button("ทำนายยอดขาย (Predict)"):
    input_df = pd.DataFrame({
        'Platform': [platform],
        'Genre': [genre],
        'Publisher': [publisher],
        'Year': [year]
    })
    
    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("🎯 ผลการคาดการณ์")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("Estimated Global Sales", f"{prediction:.2f}M Units")
        with col_res2:
            st.success(f"เกมแนว **{genre}** บนเครื่อง **{platform}** คาดว่าจะมียอดขายประมาณ **{prediction:.2f}** ล้านชุดทั่วโลก")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดตอนทำนาย: {e}")

# --- 5. Insights ---
st.markdown("---")
st.header("📊 Model Insights & Analytics")

tab1, tab2 = st.tabs(["🎯 ความแม่นยำ (Metrics)", "🧬 ปัจจัยสำคัญ (Feature Importance)"])

with tab1:
    st.subheader("ประสิทธิภาพของโมเดล")
    m = metadata.get('metrics', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score (ความแม่นยำ)", f"{m.get('r2', 0):.4f}")
    col2.metric("MAE (ค่าความคลาดเคลื่อน)", f"{m.get('mae', 0):.4f}")
    col3.metric("RMSE", f"{m.get('rmse', 0):.4f}")
    st.info("💡 ค่า R² ยิ่งใกล้ 1.00 ยิ่งแม่นยำ")

with tab2:
    st.subheader("ปัจจัยที่มีผลต่อยอดขายมากที่สุด")
    top_features_data = metadata.get('top_features', [])
    if top_features_data:
        df_importance = pd.DataFrame(top_features_data)
        chart_data = df_importance.head(10).set_index('feature')
        st.bar_chart(chart_data['importance'])
        st.write("กราฟแสดงค่าความสำคัญของแต่ละปัจจัยที่โมเดลใช้ในการคำนวณ")
    else:
        st.warning("ไม่พบข้อมูล Feature Importance")