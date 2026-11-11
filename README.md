# Formação de times em PjBL com equidade e memória entre módulos

Código, dados e resultados do artigo **“Otimização da formação de times em PjBL com
restrição de equidade e memória entre módulos”**.

O repositório compara sete políticas de formação de times ao longo de oito módulos de um currículo
de Aprendizagem Baseada em Projetos (PjBL), usando dados reais de 25.354 estudantes do Open University
Learning Analytics Dataset, e traz uma ferramenta pronta para formar times em turmas reais.

*English summary.* Code, data and results for a study of team formation across a project-based
curriculum. Seven policies are compared over eight modules on cohorts drawn from 25,354 students of the
Open University Learning Analytics Dataset. A simulated-annealing scheduler that balances skill and
engagement, never leaves an under-represented student alone in a team and remembers past pairings keeps
balance high while giving students 21 to 24 distinct teammates out of 24. The repository also ships a
command-line tool to form teams for a real class roster.

## Principais resultados

| | Turma de 40 | Turma de 120 | Turma de 400 |
|---|---|---|---|
| Sub-representados isolados pelo sorteio (1º módulo) | 55,4% | 51,1% | 59,4% |
| Equilíbrio mantido, annealing sem / com equidade | 90,9% / 82,7% | 93,9% / 90,3% | 91,2% / 88,7% |
| Duplas repetidas (módulos 2–8): sorteio | 26,4% | 9,4% | 3,0% |
| Duplas repetidas: annealing + equidade, refeito a cada módulo | 48,6% | 18,5% | 7,0% |
| Duplas repetidas: annealing + equidade + memória | 14,8% | 1,6% | 0,2% |
| Colegas distintos após 8 módulos (máx. 24), com memória | 20,9 | 23,7 | 24,0 |

Todos os números do artigo, com intervalos de confiança, estão em
[`results/summary/paper_numbers.md`](results/summary/paper_numbers.md).

## Estrutura

```
├── src/teamform/            biblioteca
│   ├── algorithms.py        sorteio, serpentina e simulated annealing (equilíbrio + equidade + memória)
│   ├── curriculum.py        políticas aplicadas ao longo de vários módulos
│   ├── metrics.py           equilíbrio, isolamento, duplas repetidas
│   ├── data.py              leitura dos dados e sorteio de turmas
│   └── api.py               form_teams(): uso em turmas reais
├── scripts/
│   ├── 01_prepare_data.py   reconstrói data/processed/students.csv a partir do OULAD
│   ├── 02_run_experiments.py  roda todos os experimentos do artigo
│   ├── 03_analyze.py        consolida resultados e calcula os números do artigo
│   ├── 04_make_figure.py    gera a Figura 1
│   └── form_teams.py        forma times para uma turma a partir de um CSV
├── data/processed/students.csv   dados usados (derivados do OULAD, CC BY 4.0)
├── results/raw/             resultados brutos de todas as simulações
├── results/summary/         tabelas-resumo e números do artigo
├── figures/                 Figura 1 do artigo
├── examples/                turma fictícia para testar a ferramenta
└── tests/                   testes automatizados
```

## Instalação

Requer Python 3.9 ou superior.

```bash
git clone <url-do-repositorio>
cd pjbl-team-formation
python -m venv .venv && source .venv/bin/activate      # opcional
pip install -r requirements.txt
make test                                               # ou: python -m pytest -q tests
```

## Reproduzir o artigo

Os resultados já estão no repositório. Para refazê-los do zero:

```bash
python scripts/02_run_experiments.py    # cerca de 5 minutos em um núcleo de CPU
python scripts/03_analyze.py            # gera results/summary/ e paper_numbers.md
python scripts/04_make_figure.py        # gera figures/figura1.png
```

ou simplesmente `make all`. As sementes aleatórias são fixas, e os resultados saem idênticos aos
publicados. O teste `test_data_and_exact_reproduction_of_published_results` confere isso.

Para rodar em partes, use `--only` e `--reps`, por exemplo
`python scripts/02_run_experiments.py --only 400x4 --reps 0 25`.
Para um teste rápido, use `--quick`.

## Usar com a sua turma

Prepare um CSV com uma linha por estudante: um identificador, uma medida de desempenho (por exemplo,
a primeira nota), opcionalmente uma medida de engajamento e colunas 0/1 para grupos sub-representados.

```bash
# primeiro módulo
python scripts/form_teams.py --roster examples/roster_example.csv --id-col student \
    --skill-col grade --engagement-col activity --group-cols woman disability \
    --team-size 4 --out times_modulo1.csv

# módulos seguintes: passe os times anteriores para evitar repetir duplas
python scripts/form_teams.py --roster examples/roster_example.csv --id-col student \
    --skill-col grade --engagement-col activity --group-cols woman disability \
    --team-size 4 --history times_modulo1.csv --out times_modulo2.csv
```

Em Python:

```python
import sys; sys.path.insert(0, "src")      # ou: pip install -e .
from teamform import form_teams

times = form_teams(skill=notas, engagement=atividade, groups=grupos,
                   team_size=4, previous_assignments=[times_anteriores])
```

Se o número de estudantes não for múltiplo do tamanho do time, alguns times ficam com um membro a
menos. As opções `--no-fairness` e `--no-memory` desligam a restrição de equidade e a memória.

## Políticas

| Código | Nome no artigo | Descrição |
|---|---|---|
| `RND` | Sorteio | novo sorteio a cada módulo |
| `SNK` | Serpentina | draft em serpentina por desempenho (gera sempre os mesmos times) |
| `SA` | Annealing (equilíbrio) | minimiza o desequilíbrio, refeito a cada módulo |
| `SA-E` | Annealing + equidade | idem, penalizando times com um único membro de um grupo |
| `SA-EM` | Annealing + equidade + memória | idem, penalizando duplas que já trabalharam juntas |
| `SA-EM-ALT` | Equidade alternada | `SA-EM` com equidade só nos módulos ímpares |
| `SA-EM-ADAPT` | Equidade adaptativa | `SA-EM` sem a regra para grupos com 2 ou 3 membros na turma |

## Modelo

Cada estudante tem desempenho *s* e engajamento *e*, padronizados na turma. O annealing minimiza,
por trocas de pares,

```
Σ_times [ (média_s − média_turma_s)² + (média_e − média_turma_e)² ]
  + λ · nº de (time, grupo) com exatamente um membro do grupo        (equidade, λ = 1)
  + μ · Σ_duplas no mesmo time  nº de vezes que a dupla já trabalhou junta   (memória, μ = 0,1)
```

Os objetivos são combinados por soma ponderada, com pesos fixos.

## Arquivos de resultados

* `results/raw/all_modules.csv`: uma linha por turma × política × módulo (equilíbrio, isolamento por
  grupo, duplas repetidas, tempo de execução).
* `results/raw/all_curriculum.csv`: uma linha por turma × política com o resumo dos 8 módulos (colegas
  distintos, déficit dos protegidos, modularidade da rede, isolamento acumulado, previsão hipergeométrica).
* `results/summary/cohort_summary.csv`: junção das duas, com o equilíbrio relativo ao sorteio.
* `results/summary/policy_summary.csv`: médias por tamanho de turma, tamanho de time e política.

## Dados e licença

O código está sob a licença MIT. Os dados em `data/processed/` derivam do OULAD (CC BY 4.0). Veja
[`data/README.md`](data/README.md) para a descrição das colunas e a fonte.
