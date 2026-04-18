EDA output folder — purpose of each file
        ==========================================

        CSV (tables for the report)
        -----------------------------
        eda_data_quality.csv          Raw vs finished; missing grade/time on finished rows
        eda_numeric_summary.csv      Grade & time distribution stats; pass-rate style columns
        eda_student_attempts.csv     Re-attempt rates (supports H3, H7, H9)
        eda_first_to_last_delta.csv  Mean/median change last−first grade for multi-attempters
        eda_cross_quiz_students.csv  Overlap of student IDs across quiz files

        PNG (figures)
        -------------
        eda_grade_histograms.png           Grade mass per quiz (+ mean/median lines)
        eda_grade_ecdf_by_quiz.png        Cumulative grades — compare quiz difficulty at once
eda_time_histograms.png          Cross-quiz time: boxplots (median, IQR); 99.5th pctile cap
eda_time_boxplot_by_quiz.png     Same data, horizontal boxplot layout
        eda_attempts_per_student.png      How many students take 1, 2, … attempts
        eda_first_to_last_grade_delta.png Improvement distribution for retakers
        eda_question_mean_marks.png       Mean raw marks per question (H2)
        eda_question_difficulty_heatmap.png Normalised means — hard vs easy items by quiz
        eda_grade_vs_time_hexbin.png      Density of grade vs time (H1 / H5)
        eda_attempts_timeline.png          Daily attempt volume (H8)
        eda_best_grade_by_max_attempts.png Exploratory link attempts → best score (H9)