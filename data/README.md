# Dados

## `processed/students.csv`

Tabela usada em todos os experimentos: os 25.354 estudantes do Open University Learning
Analytics Dataset (OULAD) que têm nota na primeira avaliação da disciplina, em 22 ofertas
de 7 disciplinas.

| coluna | descrição |
|---|---|
| `code_module`, `code_presentation` | disciplina e oferta (a turma de origem) |
| `id_student` | identificador anonimizado do OULAD |
| `gender` | `F` ou `M` |
| `disability` | deficiência declarada (`True`/`False`) |
| `imd_band` | faixa do Index of Multiple Deprivation da área de residência |
| `final_result` | resultado final na disciplina |
| `score` | nota na primeira avaliação não final da oferta (desempenho) |
| `clicks28` | cliques na plataforma nos primeiros 28 dias (engajamento) |

A ordem das linhas importa: os coortes são sorteados por posição, então mantenha o arquivo
como está para reproduzir exatamente os resultados.

Para reconstruir o arquivo do zero:

```bash
python scripts/01_prepare_data.py --source github          # baixa o espelho oficial em R (requer pyreadr)
python scripts/01_prepare_data.py --source csv --csv-dir caminho/para/csvs   # CSVs do site oficial
```

## Fonte e licença

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
*Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

O OULAD é distribuído sob a licença CC BY 4.0 (https://analyse.kmi.open.ac.uk/open_dataset).
Os dados são públicos e anonimizados, e nenhum participante foi contatado.
