# photohack
PhotoHack Hackaton

Docker
------

```bash
$ docker build -t photohack .

$ sudo docker run -d --rm --name photohack -p 80:80 \
-v /home/decaz89/2019-photohack/models:/usr/src/app/models photohack
```
