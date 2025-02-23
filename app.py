import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import joblib
# 🎨 Cấu hình giao diện
st.set_page_config(page_title="Superstore Analysis & Prediction", page_icon="📊", layout="wide")

# 🛠 Load dữ liệu có cache
@st.cache_data
def load_data():
    return pd.read_csv("saukhichinhsua.csv", encoding="ISO-8859-1")

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
    st.stop()


# 🔄 Khởi tạo session ID để tránh trùng lặp ID của Streamlit
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# ========================== HÀM VẼ BIỂU ĐỒ ==========================

# 🔹 Histogram
def plot_univariate(col):
    fig = px.histogram(df, x=col, nbins=20, title=f"📊 Phân bố của {col}",
                       opacity=0.8, text_auto=True, template="plotly_dark",
                       color_discrete_sequence=["#FF4C4C"])
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{st.session_state.run_id}")

# 🔹 Boxplot
def plot_boxplot(col):
    fig, ax = plt.subplots(figsize=(10, 5))
    with plt.style.context("ggplot"):
        sns.boxplot(x=df[col], ax=ax, color="royalblue")
        ax.set_title(f"📦 Boxplot của {col}", fontsize=14)
    st.pyplot(fig)

# 🔹 Histogram + KDE
def plot_histogram_kde(col):
    fig = px.histogram(df, x=col, nbins=30, histnorm="probability density",
                       title=f"📈 Phân bố và mật độ KDE của {col}",
                       opacity=0.6, template="plotly_dark", color_discrete_sequence=["blue"],
                       marginal="violin")
    st.plotly_chart(fig, use_container_width=True, key=f"kde_{col}_{st.session_state.run_id}")

# 🔹 Phân tích 2 biến
def plot_two_variables(df, col1, col2_value, plot_type="bar"):
    """Vẽ biểu đồ giữa một biến bất kỳ và trạng thái cụ thể của loan_status."""
    st.subheader(f"📊 {col1} theo '{col2_value}' trong Loan Status")

    if col1 not in df.columns:
        st.error(f"⚠️ Cột `{col1}` không tồn tại trong dữ liệu!")
        return

    if "loan_status" not in df.columns:
        st.error(f"⚠️ Cột `loan_status` không tồn tại trong dữ liệu!")
        return

    filtered_df = df[df["loan_status"] == col2_value]
    if filtered_df.empty:
        st.warning(f"⚠️ Không có dữ liệu nào với trạng thái '{col2_value}'!")
        return

    key_id = f"{col1}_{col2_value}_{plot_type}_{st.session_state.run_id}"

    if plot_type == "bar":
        count_data = filtered_df[col1].value_counts(normalize=True) * 100
        fig = px.bar(x=count_data.index, y=count_data.values,
                     labels={'x': col1, 'y': 'Tỷ lệ (%)'},
                     title=f"Tỷ lệ phần trăm của {col1} theo '{col2_value}'",
                     color_discrete_sequence=["#636EFA"])
        fig.update_layout(xaxis_tickangle=-45)
    elif plot_type == "box":
        if pd.api.types.is_numeric_dtype(df[col1]):
            fig = px.box(filtered_df, x="loan_status", y=col1,
                         title=f"Boxplot của {col1} theo '{col2_value}'",
                         color="loan_status",
                         color_discrete_sequence=["#EF553B"])
        else:
            st.error("⚠️ Boxplot chỉ áp dụng cho biến số!")
            return
    else:
        st.error("⚠️ Kiểu biểu đồ không hợp lệ!")
        return

    st.plotly_chart(fig, key=key_id)

# ========================== GIAO DIỆN STREAMLIT ==========================

# 🎯 Tạo Tabs
tab1, tab2, tab3 = st.tabs(["📊 Phân tích dữ liệu","📊Phân tích 2 biến" ,"🔢 Dự đoán số lượng bán"])

# ===================== Tab 1: Phân Tích Dữ Liệu =====================
with tab1:
    st.title("📊 Phân Tích Dữ Liệu")
    st.markdown("### 🔍 Chọn một cột để phân tích:")

    # Chọn cột dữ liệu
    column_selected = st.selectbox("📝 Chọn cột:", df.columns)

    # Hiển thị thông tin cột
    st.markdown(f"## 📌 Thông tin về `{column_selected}`")
    if df[column_selected].dtype in ['int64', 'float64']:
        st.write(df[column_selected].describe())
    else:
        st.write("🔹 Giá trị duy nhất:", df[column_selected].unique())
        st.write("🔹 Số lượng mỗi giá trị:", df[column_selected].value_counts())

    # Hiển thị biểu đồ
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Histogram")
        plot_univariate(column_selected)

    with col2:
        st.markdown("### 📦 Boxplot")
        plot_boxplot(column_selected)

    st.markdown("### 📈 KDE Density Plot")
    plot_histogram_kde(column_selected)

# ===================== Tab 2: Phân Tích Hai Biến =====================
with tab2:
    st.title("🔢 Phân Tích Hai Biến (Với Loan Status)")

    # Chọn biến đầu tiên
    col1_name = st.selectbox("📌 Chọn cột 1:", df.columns, key="col1")

    # Chọn biến thứ hai trong 'loan_status'
    if "loan_status" in df.columns:
        loan_status_values = df["loan_status"].unique()
        col2_name = st.selectbox("📌 Chọn Loan Status:", loan_status_values, key="col2")
    else:
        st.error("⚠️ Không tìm thấy cột `loan_status`!")
        st.stop()

    # Hiển thị thống kê
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### 📌 Thống kê `{col1_name}`")
        if df[col1_name].dtype in ["int64", "float64"]:
            st.write(df[col1_name].describe())
        else:
            st.write("🔹 Giá trị duy nhất:", df[col1_name].unique())
            st.write("🔹 Số lượng mỗi giá trị:", df[col1_name].value_counts())

    with col2:
        st.write(f"### 📌 Thống kê `{col2_name}` trong Loan Status")
        subset_df = df[df["loan_status"] == col2_name]
        st.write(subset_df.describe())

    # Hiển thị biểu đồ
    st.markdown("### 📊 Biểu Đồ Phân Tích Hai Biến")
    plot_two_variables(df, col1_name, col2_name, plot_type="bar")
# Cấu hình cách tải mô hình đúng
@st.cache_resource  # Lưu mô hình trong cache
def load_model(model_path="XG_Boost.pkl"):
    try:
        # Đảm bảo mở file một cách chính xác
        model = joblib.load(model_path)
        st.success("✅ Đã tải mô hình thành công!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file mô hình tại đường dẫn: {model_path}")
        st.info("📁 Vui lòng đảm bảo file mô hình 'random_forest.pkl' đã được tải lên thư mục hiện tại.")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
        st.info("⚠️ Đây có thể do phiên bản pickle hoặc sklearn không tương thích.")
        return None

# Tải mô hình
model = load_model()

# Debug mô hình
if 'model' in locals():
    st.sidebar.write(f"Model type: {type(model)}")
else:
    st.sidebar.error("Model không được định nghĩa")

# Xác định các cột số và cột phân loại (cần giống như lúc train)
numeric_cols = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
    'annual_inc',  'dti', 'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'collections_12_mths_ex_med'
]
categorical_cols = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'issue_d', 'purpose',  'addr_state', 'earliest_cr_line'
]

# Hàm xử lý cột earliest_cr_line
def extract_year(value):
    try:
        return datetime.now().year - int(value[-2:])  # Lấy 2 số cuối (ví dụ: '99' -> 1999)
    except:
        return None  # Gán None nếu lỗi

# Hàm xử lý dữ liệu đầu vào giống như khi huấn luyện
def preprocess_input(df):
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    df_processed = df.copy()

    # Xử lý trường earliest_cr_line
    df_processed["earliest_cr_line"] = df_processed["earliest_cr_line"].apply(extract_year)

    # Chuẩn hóa dữ liệu số
    scaler = MinMaxScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    # Mã hóa dữ liệu danh mục
    encoder = OrdinalEncoder()
    df_processed[categorical_cols] = encoder.fit_transform(df_processed[categorical_cols])

    # Bổ sung cột thiếu (nếu cần) - dựa vào X_train_final.shape[1] = 23
    # Thêm cột giả nếu thiếu
    expected_cols = 23
    current_cols = len(df_processed.columns)

    if current_cols < expected_cols:
        missing_cols = expected_cols - current_cols
        for i in range(missing_cols):
            col_name = f"feature_{current_cols + i}"
            df_processed[col_name] = 0  # Giá trị mặc định

    # Đảm bảo thứ tự cột đúng
    # Lưu ý: Bạn nên thay thế dòng dưới đây với thứ tự chính xác của các cột trong tập huấn luyện nếu biết
    # df_processed = df_processed[feature_names]  # feature_names là danh sách thứ tự cột chính xác

    return df_processed

with tab3:
    st.header("🔮 Dự đoán Khoản Vay")

    # Hiển thị debug info về model
    if model is not None:
        st.sidebar.success("✅ Mô hình đã được tải")
        if hasattr(model, 'feature_names_in_'):
            st.sidebar.info(f"Số lượng đặc trưng mô hình: {len(model.feature_names_in_)}")
            st.sidebar.info(f"Tên các đặc trưng: {', '.join(model.feature_names_in_)}")

    # Nhập dữ liệu từ người dùng
    st.write("🔹 **Nhập thông tin để dự đoán**")
    user_input = {
        # Các cột số
        "loan_amnt": st.number_input("Số tiền vay", min_value=500, max_value=50000, value=10000),
        "funded_amnt": st.number_input("Số tiền được cấp vốn", min_value=500, max_value=50000, value=10000),
        "funded_amnt_inv": st.number_input("Số tiền nhà đầu tư cấp vốn", min_value=500, max_value=50000, value=10000),
        "term": st.number_input("Thời hạn khoản vay (tháng)", min_value=12, max_value=60, value=36),
        "int_rate": st.slider("Lãi suất (%)", 5.0, 30.0, 12.5),
        "annual_inc": st.number_input("Thu nhập hàng năm ($)", value=50000),
        "dti": st.number_input("Tỷ lệ nợ trên thu nhập (%)", value=20.0),
        "delinq_2yrs": st.number_input("Số lần trễ hạn trong 2 năm", min_value=0, max_value=10, value=0),
        "inq_last_6mths": st.number_input("Số lần truy vấn tín dụng trong 6 tháng", min_value=0, max_value=10, value=1),
        "open_acc": st.number_input("Số tài khoản đang mở", min_value=0, max_value=50, value=10),
        "pub_rec": st.number_input("Số lần bị báo cáo công khai", min_value=0, max_value=10, value=0),
        "revol_bal": st.number_input("Số dư quay vòng", value=10000),
        "revol_util": st.slider("Sử dụng tín dụng (%)", 0.0, 100.0, 50.0),
        "total_acc": st.number_input("Tổng số tài khoản tín dụng", value=15),
        "collections_12_mths_ex_med": st.number_input("Số lần thu hồi nợ trong 12 tháng qua", min_value=0, max_value=10, value=0),

        # Các cột phân loại (chọn từ giá trị unique trong dataset)
        "grade": st.selectbox("Xếp hạng tín dụng", df["grade"].unique()),
        "sub_grade": st.selectbox("Xếp hạng phụ", df["sub_grade"].unique()),
        "home_ownership": st.selectbox("Quyền sở hữu nhà", df["home_ownership"].unique()),
        "verification_status": st.selectbox("Trạng thái xác minh", df["verification_status"].unique()),
        "issue_d": st.selectbox("Ngày cấp khoản vay", df["issue_d"].unique()),
        "purpose": st.selectbox("Mục đích vay", df["purpose"].unique()),
        "addr_state": st.selectbox("Bang cư trú", df["addr_state"].unique()),
        "earliest_cr_line": st.text_input("Năm mở tài khoản tín dụng (MM-YY)", "Jan-99"),
    }


    # Nút dự đoán với xử lý lỗi
    if st.button("🔮 Dự đoán"):
        if model is not None:
            try:
                # Hiển thị trạng thái xử lý
                with st.spinner("Đang xử lý dữ liệu..."):
                    # Chuyển dict thành DataFrame
                    user_df = pd.DataFrame([user_input])

                    # Debug dữ liệu đầu vào
                    st.write("Số lượng cột đầu vào:", len(user_df.columns))

                    # Chuẩn hóa dữ liệu trước khi dự đoán
                    processed_input = preprocess_input(user_df)

                    # Debug dữ liệu đã xử lý
                    st.write("Số lượng cột sau xử lý:", len(processed_input.columns))

                    # Dự đoán
                    prediction = model.predict(processed_input)

                    # Hiển thị kết quả
                    st.success(f"### 🎯 **Kết quả dự đoán:** {prediction[0]}")

                    # Hiển thị xác suất nếu mô hình hỗ trợ
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(processed_input)
                        st.write("### 📊 **Xác suất dự đoán:**")
                        for i, prob in enumerate(proba[0]):
                            st.write(f"Lớp {model.classes_[i]}: {prob*100:.2f}%")
            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
                st.info("🛠️ Chi tiết debug:")
                st.exception(e)
                st.info("💡 Gợi ý: Đảm bảo cấu trúc dữ liệu đầu vào khớp với dữ liệu huấn luyện")
        else:
            st.error("❌ Không thể dự đoán vì mô hình chưa được tải.")
            st.info("📁 Vui lòng kiểm tra file mô hình 'XG_Boost.pkl'.")