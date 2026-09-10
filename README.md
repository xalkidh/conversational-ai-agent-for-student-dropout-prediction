# Conversational AI Agent for Student Dropout Prediction

## 1. System Overview

This project implements a domain-aware conversational AI agent that combines a Retrieval-Augmented Generation (RAG) system with the trained machine learning model from HW1. The agent can answer natural language questions about student dropout and make predictions for individual student profiles. It is built using LangGraph, LangChain, and FastAPI, and is powered by the Google Gemini API.

**Domain:** Student Dropout Prediction  
**Best model:** Random Forest Classifier

---

## 2. Architecture

The agent is built with LangGraph and has access to three tools:

| Tool | Description |
|------|-------------|
| `retrieve_information` | Searches the RAG knowledge base to answer factual/conceptual questions about student dropout |
| `predict_dropout` | Uses the HW1 Random Forest model to predict dropout probability for a given student profile |
| `dataset_statistics` | Returns summary statistics for any numerical column in the HW1 dataset |

I have also implemented the **dataset_statistics** tool, as I considered it the most useful addition for the student dropout domain. This tool provides key insights into the dataset, helping users better understand the data distribution and the factors that may influence dropout predictions.

### LangGraph Graph Structure

The agent flow follows these steps:

1. The user message is passed to the **agent node** (Gemini LLM)
2. The agent autonomously decides which tool to call based on the message content
3. If a tool is required, the **tool node** executes it and returns the result to the agent
4. The agent formulates the final response and returns it to the user

Conversation memory is maintained within a session via the `session_id` parameter.

---

## 3. Knowledge Base

Eight documents were selected to cover the full spectrum of factors related to student dropout, from individual academic and psychological factors, to institutional responses and prevention strategies. Together they provide the agent with a comprehensive knowledge base that mirrors the features used in the HW1 prediction model (GPA, stress, attendance, financial support, etc.).

The knowledge base consists of 8 documents stored in `data/documents/`:

| File | Content | Source |
|------|---------|--------|
| `dropout_factors.txt` | Key factors associated with student dropout | https://www.nature.com/articles/s41599-025-04698-y/tables/10 |
| `dropout_factors2.txt` | Additional research on dropout risk factors | https://toknowpress.net/ISBN/978-961-6914-31-4/104.pdf |
| `dropout_prevention.txt` | Strategies and programs for dropout prevention | https://www.expandinglearning.org/expandingminds/article/proven-solution-dropout-prevention-expanded-learning-opportunities |
| `reducing_dropout_rates.txt` | Approaches to reducing dropout rates | https://riunet.upv.es/server/api/core/bitstreams/2d63098a-6701-483b-a6dc-d2fde3c88774/content |
| `stress_and_mental_health.txt` | Impact of stress and mental health on dropout | https://www.researchgate.net/publication/377873074_Factors_Affecting_Student_Drop-Out_Behavior_A_Systematic_Review_Article_Info_ABSTRACT |
| `student_housing_and_dropouts.txt` | Relationship between student housing and dropout | https://eva.dk/Media/638543977618694125/Bilag%20-%20English%20summary%20Student%20housing%20and%20dropout%20in%20higher%20education.pdf |
| `student_retention.txt` | University retention strategies | https://www.researchgate.net/publication/318133592_Student_Retention_in_Higher_Education_A_Shared_Issue |
| `wikipedia_dropping_out.txt` | Wikipedia overview of student dropout | https://en.wikipedia.org/wiki/Dropping_out |

The RAG system can answer questions such as:
- "What are the main factors that cause students to drop out?"
- "How does stress affect academic performance?"
- "What strategies do universities use to improve student retention?"

---

## 4. Student Dropout Prediction Model Integration

The prediction tool wraps the **Random Forest Classifier** trained in the Student Dropout Prediction repository. It loads `models/best_model.pkl` and `models/scaler.pkl` and applies the exact same preprocessing pipeline:

- **Feature Engineering:** Creates `study_attendance_ratio` and `stress_gpa_ratio` features
- **One-Hot Encoding:** Encodes categorical features
- **Standard Scaling:** Normalizes features using the saved scaler from HW1

### Required Input Fields:

| Field | Type | Description |
|-------|------|-------------|
| `Age` | Number | Student age |
| `Gender` | Text | Gender (Male/Female) |
| `Year_of_Study` | Number | Year of study |
| `GPA` | Number | Grade point average |
| `Attendance_Rate` | Number | Attendance percentage (0-100) |
| `Study_Hours_per_Day` | Number | Study hours per day |
| `Stress_Index` | Number | Stress index (0-10) |
| `Financial_Support` | Text | Financial support (Yes/No) |
| `Part_Time_Job` | Text | Part-time employment (Yes/No) |
| `Family_Income_Level` | Text | Family income level (Low/Medium/High) |
| `Internet_Access` | Text | Internet access (Yes/No) |

**Output:** Prediction (Dropout / No Dropout) with probability percentage.

---

## 5. Example Conversations

### Example 1 – RAG Retrieval
**User:** What are the main factors that cause students to drop out?

**Agent:** The factors that contribute to student dropout are multifaceted and often interconnected. The primary drivers typically include Academic Performance (low GPA or failing grades), Financial Constraints (lack of financial support or need to work long hours), Attendance and Engagement (low attendance rates), Stress and Mental Health (high stress, anxiety, or burnout), and Lack of Institutional Support (insufficient access to academic advising or counseling).

---

### Example 2 – Prediction
**User:** Predict dropout for a student with GPA 2.1, Attendance Rate 55, Study Hours 2, Stress Index 8, Age 20, Gender Male, Year of Study 2, Financial Support No, Part Time Job Yes, Family Income Level Low, Internet Access Yes

**Agent:** Based on the data provided, the analysis indicates a high risk of dropout (66.0%). Key contributing factors include low attendance (55%), low GPA (2.1), high stress (8/10), and financial/work pressure.

---

### Example 3 – Dataset Statistics
**User:** What is the average GPA of students in the dataset?

**Agent:** The average GPA of students in the dataset is 2.31.

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10+
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/xalkidh/hw2.git
cd hw2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 4. Run the application
python main.py
```

On first run, the application will:
- Automatically build the vector store from the documents in `data/documents/`
- Start the FastAPI server at `http://127.0.0.1:8000`
- Serve the Swagger UI at `http://127.0.0.1:8000/docs`

On subsequent runs, the vector store is loaded automatically without rebuilding.

---

## 7. Example API Call

### Standard endpoint
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main causes of student dropout?", "session_id": "user_001"}'
```

### Streaming endpoint
```bash
curl -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main causes of student dropout?", "session_id": "user_001"}'
```