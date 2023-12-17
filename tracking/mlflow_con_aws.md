# Basic AWS setup

Acá te enseñaré a configurar la instancia, db y bucket para configurar mlflow.

1. Lo primero, debes de [tener una cuenta en aws](https://aws.amazon.com/free).

2. Abre un instancia en EC2. 

Seleccionamos una instancia de  (`Amazon Linux 2 AMI (HVM) - Kernel 5.10, SSD Volume Type`) con tipo `t2.micro`, las cuales son gratuitos.


<img src="../images/ec2_os.png" width=400/>

Adicionalmente, debemos crear una nueva key la cual nos va a permitir ingresar a la instancia a través de llaves SSH. Haz click en "Create new key pair", selecciona RSA en Key pair type y .pem en el formato para la llave privada. 


Posteriormente, tenemos que editar los grupos de seguridad para que nuestra instancia EC2 pueda aceptar la conexión SSH (mediante port 22) y para mlflow nos podamos conectar a traves del port 5000 (Type Custom TCP):

<img src="../images/security_group.png" width=400/>

3. Creamos bucket para almacenar nuestros metadatos y artifacts.

Vamos a s3 damos click en "create bucket". 

Note: El nombre del bucket debe ser único. 

4. Creamos un PostgreSQL database para ser usado como backend store. 

Ingresamos al servicio de RDS damos click en "Create database". Elige la versión.

<img src="../images/postgresql.png" width=400/>

Asigna nombre a la  DB instance, asigna como master username a "mlflow" y genera la contraseña automáticamente. 

<img src="../images/db_settings.png" width=400/>

Finalmente, dada la configuración especificada, se creará una db inicial para ti. 

<img src="../images/db_configuration.png" width=400/>

Una vez creada la db, copia la contraseña porque la verás una única vez y la necesitaremos más adelante. 



A modo de resumen toma nota de lo siguiente porque lo vas a necesitar :

* master username
* password 
* initial database name
* endpoint

Once the DB instance is created, go to the RDS console, select the new db and under "Connectivity & security" select the VPC security group. Modify the security group by adding a new inbound rule that allows postgreSQL connections on the port 5432 from the security group of the EC2 instance. This way, the server will be able to connect to the postgres database.

<img src="../images/postgresql_inbound_rule.png" width=400/>

5. Conexión a la EC2 instancia y acceder al server tracking de mlflow. 

Vamos a darle vida y setear dependencias y todo lo que necesitamos:

* `sudo yum update`
* `pip3 install mlflow boto3 psycopg2-binary`
* `aws configure`   # you'll need to input your AWS credentials here
* `mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME`

Note: before launching the server, check that the instance can access the s3 bucket created in the step number 3. To do that, just run this command from the EC2 instance: `aws s3 ls`. You should see the bucket listed in the result.

6. Acceso al servidor de forma local.

Abre tu navegador e ingresa: `http://<EC2_PUBLIC_DNS_DE_TU_EC2>:5000` (Esto lo puedes ver en la configuración de tu instancia).