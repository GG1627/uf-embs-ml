# UF EMBS Event Attendance Prediction Model

This project builds a baseline machine learning model to predict attendance
for UF EMBS events using historical event data.

The goal is to identify patterns in event timing, type, and incentives
to help officers better plan future events.

## Getting Started

### Prerequisites
- Python 3.x
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GG1627/uf-embs-ml
   cd uf-embs-ml
   ```

### Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**
   ```bash
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Create a .gitignore file**
   - Create a file named `.gitignore` in the root directory
   - Add the following line to it:
     ```
     venv/
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Used

The model is trained on past UF EMBS event data, including:
- Event type (workshop, GBM, competition, etc.)
- Date and start time
- Points offered
- Food availability
- Virtual vs in-person
- Total recorded attendance (target variable)

## Contributors

- **Gael Garcia**