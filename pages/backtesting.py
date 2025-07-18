import streamlit as st

def main():
    st.title("📈 Backtesting Test")
    st.write("This is a test page to verify navigation works.")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")

if __name__ == "__main__":
    main()
