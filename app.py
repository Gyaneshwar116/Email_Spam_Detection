import streamlit as st
import joblib
import sklearn

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("Email_spam_model.pkl")

# ----------------------------
# Application Header
# ----------------------------
st.title("📧 Email Spam Detection")

st.caption(
    "Machine Learning email classification"
)

st.divider()


# ----------------------------
# Email Input
# ----------------------------
st.subheader("Analyze Your Email")

email = st.text_area(
    "Enter Email Text",
    height=180,
    placeholder="Type or paste your email message here..."
)


# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict", type="primary", use_container_width=True):

    if not email.strip():
        st.warning("Please enter an email message.")

    else:
        prediction = model.predict([email])[0]
        probabilities = model.predict_proba([email])[0]

        classes = list(model.classes_)

        spam_index = classes.index(1)
        ham_index = classes.index(0)

        spam_probability = probabilities[spam_index] * 100
        ham_probability = probabilities[ham_index] * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Spam Email")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

            st.progress(float(spam_probability / 100))

        else:
            st.success("✅ Ham — Not Spam")

            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )

            st.progress(float(ham_probability / 100))

        with st.expander("View Detailed Probabilities"):

            col1, col2 = st.columns(2)

            col1.metric(
                "Spam",
                f"{spam_probability:.2f}%"
            )

            col2.metric(
                "Ham (Not Spam)",
                f"{ham_probability:.2f}%"
            )


# ----------------------------
# Footer
# ----------------------------
st.divider()

# st.caption(
#     "Built with Python, Scikit-learn and Streamlit" )   