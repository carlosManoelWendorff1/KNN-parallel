# KNN com OpenMP

## 🖥️ Ambiente de Execução

- **CPU**: Intel Core i5‑12400F (12ª geração, 2.50 GHz)
- **RAM**: 32 GB
- **GPU**: RTX 4060 Ti 8 GB (não utilizada nesta análise, o foco é CPU/OpenMP)
- **SO**: Windows 11 Pro (64 bits)
- **Compilador**: GCC (via WSL / MinGW / outro, conforme ambiente)

## Compilar

```bash
gcc -fopenmp knn_parallel.c -o knn_parallel
```

## Executar

```bash
knn_parallel < example.in
```

Formato do arquivo de entrada

```bash
n_groups=2
label=A
length=2
(1.0,1.0)
(2.0,2.0)
label=B
length=2
(10.0,10.0)
(11.0,11.0)
k=2
(1.5,1.5)
```

Saída esperada

```bash
=== Teste de Performance ===
Grupos: 50 | K: 200000

Vers├úo Sequencial:
  Resultado: r
  Tempo: 16.0670 segundos

Vers├úo Paralela:
Threads  Tempo (s)    Speedup    Efici├¬ncia
----------------------------------------
2        16.8940      0.95       0.48       r
4        12.1850      1.32       0.33       r
8        8.9930       1.79       0.22       r
```

O programa mostra o tempo sequencial e o tempo paralelo com 2, 4 e 8 threads.

## Perguntas e respostas

```bash
  1. Quais dados devem ser enviados pelo mestre?
  O mestre compartilha n_groups, groups, k, to_evaluate, labels e distances.

  2. Quais dados devem ser retornados pelos trabalhadores?
  Cada thread retorna sua lista local de top-k (distâncias e rótulos).

  3. Como o qsort pode ser paralelizado? Usando apenas OpenMP é suficiente?
  Sim, com OpenMP tasks. Para k pequeno não é necessário.

  4. Como avaliar o desempenho?
  Medindo tempo sequencial vs paralelo, calculando speedup = T_seq/T_par e eficiência = speedup/n_threads.

  5. Sugestão de dataset
  Usar main.py com n_groups=50, min_points=50, max_points=150, k=1000.
```

## 📈 Gráficos

Veja os gráficos na pasta results
