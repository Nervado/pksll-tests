### PLC-Tester

Este projeto corresponde a uma biblioteca que permite realizar testes nos projetos PLCS da Marcas Rockwell, Altus e Emerson das Unidades de produção de Búzios

O arquivo .env contém as variáveis para configurar o sistema quando em produção e o arquivo .env contem as variáveis para o ambiente de test/desenvolvimento.

## Ambiente de desenvolvimento

1. Necessário ter um servidor UA acessível para conexão e uma cópia da imagem docker aut/vip:0.0.1.

2. Subir o container de teste através do comando:

```bash
[user@host]$ docker-compose -f docker-compose.test.yml up -d
```

3. Acessar o container através de:

```bash
[user@host]$ docker exec -it data_logger bash
```

4. Rodar o sistema através de:

```bash
[user@host]$ data-logger
```

obs.: o comando pode ser cancelado através da combinação de teclas control + C.

5. Rodar os testes:

```bash
[user@host]$ ptw src/ -- --vvxrP
```

6. Para alterar as configurações de variaveis de ambiente disponiveis no arquivo .env.test é necessario remover o container do projeto e executar o docker compose novamente:

```bash
[user@host]$ docker-compose -f docker-compose.test.yml down && docker-compose -f docker-compose.test.yml up -d && docker exec -it data_logger bash
```

## Build do projeto

1. Para criar uma build de distribuição do projeto execute o comando:

```bash
[user@host]$ python -m build
```

2. Certifique-se de que todas as dependencias informadas no arquivo requirements.txt estão instaladas no container.

3. O arquivo de saida da build é no formato .whl com o tag de versão do programa, exemplo:

data_logger-0.0.1-py3-none.any.whl


## Ambiente de produção

0. Na máquina onde será executado o container da aplicação deverá haver uma pasta contendo o arquivo docker-compose.yml, o arquivo .env e uma pasta contendo a ultima build. Note que o arquivo .env contem uma variavel que indica a versão do programa. Caso seja realizada nova build recomenda-se a atualização da variavel VERSION para manter o tracking do versionamento correto.

1. Adicionar ao arquivo tags.csv dentro da pasta src/data_logger/config/ os tags desejados seguindo o mesmo padrão usado:

|      tag     |     io    | well-letter |   well-name   |                  path-a                  |                  path-b                  | engunit |  type | conv-arg-0 | conv-arg-1 | conv-arg-2 | conv-arg-3 | server | uep |
|:------------:|:---------:|:-----------:|:-------------:|:----------------------------------------:|:----------------------------------------:|:-------:|:-----:|:----------:|:----------:|:----------:|:----------:|:------:|:---:|
| PIT_1210007A | P-JUZ-CKP |      A      |  9-BUZ-1-RJS  |     ns=2;s=AI.PSD.AI_PV_PIT_1210007A     |     ns=2;s=AI.PSD.AI_PV_PIT_1210007A     |   kPag  | float |      0     |      1     |      0     |      1     |   PSD  | P77 |
| PIT_1210007B | P-JUZ-CKP |      B      |  9-BUZ-2-RJS  |     ns=2;s=AI.PSD.AI_PV_PIT_1210007B     |     ns=2;s=AI.PSD.AI_PV_PIT_1210007B     |   kPag  | float |      0     |      1     |      0     |      1     |   PSD  | P77 |
| PIT_1210007C | P-JUZ-CKP |      C      |  9-BUZ-3-RJS  |     ns=2;s=AI.PSD.AI_PV_PIT_1210007C     |     ns=2;s=AI.PSD.AI_PV_PIT_1210007C     |   kPag  | float |      0     |      1     |      0     |      1     |   PSD  | P77 |
| PIT_1210007D | P-JUZ-CKP |      D      |  9-BUZ-4-RJS  |     ns=2;s=AI.PSD.AI_PV_PIT_1210007D     |     ns=2;s=AI.PSD.AI_PV_PIT_1210007D     |   kPag  | float |      0     |      1     |      0     |      1     |   PSD  | P77 |


2. Configure os valores das variaveis de ambiente: PLC_URL, MAX_SAMPLES, SAMPLING_PERIOD, UEP.

3. Configure os valores das variaveis de acesso ao banco de dados: INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN:

4. Utilize o comando a seguir para iniciar o sistema:

```bash
[user@host]$ docker-compose up -d
```
5. Utilize o comando a seguir para ver o log de saída, a variavel LOG_LEVEL informa o tipo de evento que será capturado pelo log (opções são as default do python DEBUG, WARN, INFO, ERROR e CRITICAL).

```bash
[user@host]$ docker-compose logs datalogger
```

6. Para interromper o funcionamento da aplicação execute o seguinte comando:

```bash
[user@host]$ docker-compose down
```

## Cuidados!

1. Não utilize uma variável com valor muito baixo para SAMPLING_PERIOD ( minimo de 5s ).
2. Caso não seja absolutamente necessário não deixe este sistema rodando muito tempo em ambiente de produção.


obs:

@petrobr2022