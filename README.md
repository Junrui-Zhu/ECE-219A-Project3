# Project 3: Recommender Systems

## Table of Contents
1. [How to Run the Code](#how-to-run-the-code)
2. [Dependencies](#dependencies)
3. [File Structure](#file-structure)
4. [Authors](#authors)
---
## How to Run the Code
Necessary steps to run the code:

1. Navigate to the project directory:
   ```bash
   cd ECE-219A-Project3
   ```

2. Install the required dependencies. (Check #dependencies for detail)

3. There are 2 datasets in this project. In our submitted codes, only 1 dataset (movie) is included. We dropped the irrelevant .csv files and only left ratings.csv. For web10k dataset, you need to download it. After that, change the corresponding path in q_16_1.py, q15.py, q16_2.py, web10k.py so that the web10k dataset can be loaded successfully. 

4. Each python script performs a certain algorithm/task, or answers several questions by running it. Code comments are provided at the beginning of each python script.
 Question 1: proj3_q1.py
 Question 4: proj3_q4.py
 Question 6: q6.py
 Question 8: q8a.py,q8b.py,q8c.py
 Question 9: q9.py
 Question 10: q10A.py,q10b.py,q10c.py
 Question 11: naive_filter.py
 Question 12: q12.py
 Question 13: web10k.py
 Question 14: light_gbm.py
 Question 15: q15.py
 Question 16: q_16_1.py, q16_2.py
---

## Dependencies
This project requires the following libraries and tools. 
- python==3.9.13
- lightgbm>=4.0.0
- scikit-learn>=1.2.0
- numpy>=1.23.0
- pandas>=1.5.0
- matplotlib>=3.6.0
- scipy>=1.10.0
- scikit-surprise>=1.1.3
---

## File Structure
This Project is organized as follows:
```bash              
├── nndl/                 # Source code
│   ├── light_gbm.py         
│   ├── naive_filter.py
│   ├── proj3_q1.py
│   ├── proj3_q4.py
│   ├── q_16_1.py
│   ├── q6.py
│   ├── q8a.py
│   ├── q8b.py
│   ├── q8c.py
│   ├── q9.py
│   ├── q10A.py
│   ├── q10b.py
│   ├── q10c.py 
│   ├── q12.py 
│   ├── q15.py 
│   ├── q16_2.py 
│   └── web10k.py 
├── Synthetic_Movie_Lens/ 
│   ├── ratings.csv
│   └── README.txt 
├── learning_to_rank_helper.ipynb
└── README.md            # Documentation
```

Notes:
- All source code is located in the `nndl/` folder.
- Use .ipynb files to download and use the dataset.

---

## Authors

This project was collaboratively developed by the following contributors:

| Name                | UID                       |  Contact               |
|---------------------|---------------------------|------------------------|
| **LiWei Tan**       | 206530851                 | 962439602@qq.com       |
| **TianXiang Xing**  | 006530550                 | andrewxing43@g.ucla.edu|
| **Junrui Zhu**      | 606530444                 | zhujr24@g.ucla.edu     |
---
