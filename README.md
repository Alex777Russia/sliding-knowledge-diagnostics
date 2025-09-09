# Macan-Team

Introduction Hackathon for ITMO Masters - AI Talent Hub 2025-2027

# Launch


```bash
git clone https://github.com/Alex777Russia/sliding-knowledge-diagnostics.git
cd sliding-knowledge-diagnostics
```

To run a contnainer you will need to create ```.env``` file in current dir first. Write those variables:

```bash

FOLDER_ID=<your_yandex_api_folder_id>
YANDEX_API_KEY=<your_yandex_api_key>

```

Also you need a database in current dir with name  ```data.csv``` format.

Our data base can be found [here](https://drive.google.com/file/d/1X8b4qg_tjaGOvKUVgPVnCeo5sQ1XIRih/view?usp=sharing).

Start app:

```bash
bash docker/build.sh
bash docker/run.sh
```

Local app can be accessed by link ```http://0.0.0.0:7860```