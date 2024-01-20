# GPU usage in Deep Learning
```python
def gpu_activation():
    from tensorflow.config import experimental as tf_exp
    physical_devices = tf_exp.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf_exp.set_memory_growth(gpu, True)
```

# Seaborn
## Args
* **bins** : le nombre de bars possible sur l'axe des x

# Math
\begin{align*}
    A \times B &= \begin{pmatrix}
        a_{1,1}&a_{1,2}&a_{1,3}\\a_{2,1}&a_{2,2}&a_{2,3}\\a_{3,1}&a_{3,2}&a_{3,3}
    \end{pmatrix} \times \begin{pmatrix}
        b_{1,1}&b_{1,2}&b_{1,3}\\b_{2,1}&b_{2,2}&b_{2,3}\\b_{3,1}&b_{3,2}&b_{3,3}
    \end{pmatrix}\\
    &= \begin{pmatrix}
        a_{1,1} \cdot b_{1,1} + a_{1,2} \cdot b_{2,1} + a_{1,3} \cdot b_{3,1} & a_{1,1} \cdot b_{1,2} + a_{1,2} \cdot b_{2,2} + a_{1,3} \cdot b_{3,2} & a_{1,1} \cdot b_{1,3} + a_{1,2} \cdot b_{2,3} + a_{1,3} \cdot b_{3,3} \\
        a_{2,1} \cdot b_{1,1} + a_{2,2} \cdot b_{2,1} + a_{2,3} \cdot b_{3,1} & a_{2,1} \cdot b_{1,2} + a_{2,2} \cdot b_{2,2} + a_{2,3} \cdot b_{3,2} & a_{2,1} \cdot b_{1,3} + a_{2,2} \cdot b_{2,3} + a_{2,3} \cdot b_{3,3} \\
        a_{3,1} \cdot b_{1,1} + a_{3,2} \cdot b_{2,1} + a_{3,3} \cdot b_{3,1} & a_{3,1} \cdot b_{1,2} + a_{3,2} \cdot b_{2,2} + a_{3,3} \cdot b_{3,2} & a_{3,1} \cdot b_{1,3} + a_{3,2} \cdot b_{2,3} + a_{3,3} \cdot b_{3,3} \\
    \end{pmatrix} \\
\end{align*}