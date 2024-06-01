
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概要

Let's Encrypt是一个开放源代码的公共（非盈利）、自动化证书颁发机构（Certificate Authority）。该机构为互联网上的网站提供免费的SSL/TLS证书，其认证服务由一系列的ACME协议所定义。Let's Encrypt 允许任何人申请SSL/TLS证书而无需支付任何费用。

本文将详细介绍Let's Encrypt的域名验证过程。域名验证是Let's Encrypt与通用的CA机构Certificate Authority（CA）进行交互的一步，它确保了域名的所有权并对申请者的身份进行认证。

## 作者简介

李云飞，四川大学网络空间安全研究中心博士，主要研究方向为互联网安全领域的安全工程和数字经济。曾任职于阿里巴巴、腾讯等知名互联网公司，任职于四川大学网络空间安全研究中心期间为智慧通行证研究项目助理。

# 2.背景介绍

Let's Encrypt域名验证用于确认用户的服务器正在运行自己的服务且拥有指定域名的合法权益。其工作流程可以概括为以下几个步骤：

1. 用户访问其域名对应的主页，然后浏览器向Let's Encrypt的ACME服务器请求证书。
2. Let's Encrypt生成一个随机值，然后用这个随机值作为challenge并提交到域名服务器的DNS记录中。
3. 当用户的域名服务器收到Let's Encrypt的提交后，会检测到TXT记录是否存在，如果存在则验证成功，否则返回错误信息。
4. 如果域名服务器返回的TXT记录的内容与Let's Encrypt发送的challenge相符，那么就可以认为该域名属于当前用户，Let's Encrypt再向用户的服务器发出证书签名请求。
5. 用户的服务器接收到证书签名请求后，开始生成私钥和CSR文件，并把它们提交给Let's Encrypt的CA。
6. CA生成一个证书文件，然后使用私钥加密证书，并且用Let's Encrypt的公钥签署证书。
7. Let's Encrypt服务器把证书发送回用户的服务器，然后保存并安装在服务器上。

上述过程能够确保域名所有权的真实性，同时也提供了HTTPS通信的基础。 

但是，域名验证过程中还涉及一些机制，包括：

1. ACME服务器的定时任务，它会定期检查TXT记录是否已经更新或失效，如果TXT记录已失效，则会重新提交新的challenge。
2. CSR文件的生成方法，Let's Encrypt要求用户提供有效的CSR文件，才能最终生成证书。如果用户不提供合适的CSR文件，或者CSR文件中的域名与用户实际使用的域名不一致，那么证书签名请求就会失败。
3. DNS记录的生成方法，Let's Encrypt通过向域名服务器的DNS记录中提交challenge来验证域名所有权。根据规范，域名所有权必须是通过某个TXT记录来验证的。

因此，域名验证的关键在于准确的DNS记录的生成方法，否则用户可能会面临证书无法验证的问题。另外，Let's Encrypt在不同情况下的报错处理也需要考虑周全，否则可能会导致证书申请失败或其他不可预料的问题。

# 3.基本概念术语说明

## 域名验证与证书申请

首先，我们来看一下域名验证和证书申请的相关概念。

域名验证又称为通用名称认证机构（Common Name Authentication，CNAME）验证，是Let's Encrypt的域名验证环节的名字。常见的CNAME验证有DigiCert CA，Akamai Technologies，StartCom Certification Authority，Symantec Class 3 Secure Server CA等。这些认证机构的域名验证都是依靠域名服务器的DNS记录来验证域名的合法性，并不直接向证书颁发机构CA提出申请。

Let's Encrypt除了支持普通域名验证外，还支持多域名证书申请。多域名证书申请指的是同一个证书颁发机构可以为多个域名颁发证书，此时同一个证书签发机构在向认证机构的域名验证中只需要一次即可。

证书申请是指让证书颁发机构CA为一个或多个域名颁发证书。Let's Encrypt的证书申请主要由ACME协议完成，它是Let's Encrypt服务器之间通信的标准协议。目前的版本是RFC8555。ACME协议主要由两个阶段组成：

1. 注册阶段（Registration），用户需要向证书颁发机构发送注册请求，申请颁发证书的权限。
2. 认证阶段（Authorization），用户需要向证书颁发机构发送认证请求，声明自己拥有某些域名的合法权益，然后证书颁发机构根据用户提供的信息验证域名所有权，最后颁发证书。

Let's Encrypt使用ACME协议来验证域名所有权，而且目前还没有其他CA采用这种方式来验证域名。

## DNS记录类型

域名验证过程依赖域名服务器的DNS记录来实现，对于一般的域名，通常情况下都有一个A记录指向该域名的IP地址。当用户申请证书时，Let's Encrypt会在该域名下新增一条TXT记录，用来提交challenge。TXT记录的TTL通常设置为60秒到120秒之间。如果TXT记录过期或不存在，Let's Encrypt会继续尝试获取新challenge。TXT记录的内容是特定字符串（challenge），需要通过有效的CSR文件和私钥进行签名。

另一种类型的DNS记录叫做MX记录（Mail eXchanger Record），它指向邮件服务器的域名。在Let's Encrypt的申请流程中，虽然我们并不需要MX记录，但它仍然是为了验证域名所有权。如果您使用的是独立的邮件服务器，您可以在DNS设置中添加一个空的MX记录，或者指向某个不会正常解析的域名。

## DNS查询类型

为了验证域名所有权，Let's Encrypt需要向域名服务器查询TXT记录，但是对于不同类型的域名服务器，它的查询方式可能不太一样。

常见的域名服务器有BIND9、PowerDNS、Unbound、Nsd、Yadifa等。这些服务器的区别主要体现在DNS查询时的查询类型上。例如，BIND9的默认查询类型是TCP，而PowerDNS的默认查询类型是UDP。当然，用户也可以修改查询类型。

## 域名解析器（DNS Resolvers）

域名解析器（DNS resolver）是网络应用的组件之一，它负责把域名转换为IP地址。许多应用都会依赖于域名解析器才能正常运行。比如Web浏览器，电子邮箱客户端等。因此，如果域名解析器出现故障，那么这些应用也将无法正常工作。

Let's Encrypt支持两种类型的域名解析器：递归解析器和迭代解析器。

- 递归解析器（Recursive Resolver）：在域名解析时，先向本地域名服务器查询，然后从本地服务器获取结果，如没有本地服务器，则向根域名服务器查询，得到结果后，再向本地服务器查询，直到得到最终的结果。递归解析器的优点是简单易用，缺点是可能存在延迟和流量限制。
- 迭代解析器（Iterative Resolver）：迭代解析器与递归解析器的区别在于，迭代解析器在查询本地域名服务器时，不会向根域名服务器查询，而是通过配置文件找到下一步应该查询的服务器，然后再次向该服务器进行查询。迭代解析器的优点是避免了递归查询的延迟和流量限制，缺点是配置繁琐复杂。

Let's Encrypt选择了递归解析器，原因如下：

1. 浏览器等应用会优先使用递归解析器，所以Let's Encrypt尽可能保持与这些应用的兼容性。
2. 使用递归解析器的Let's Encrypt可以绕过防火墙和特殊路由器的限制，即使受限环境也是安全的。
3. 支持多种操作系统和解析器，包括Windows、Linux、MacOS、FreeBSD等。
4. 有更好的性能表现，尤其是在使用IPv6时。

Let's Encrypt的授权协议目前支持http-01和dns-01两种验证方式，这两种方式各有优劣。

- http-01方式是利用Web服务器自带的认证功能来验证域名所有权。申请证书时，Let's Encrypt会在用户的域名下设置一个HTTP路径，例如/.well-known/acme-challenge/。用户的Web服务器必须正确响应该路径，并在响应头中包含验证字符串（challenge）。当Let's Encrypt发送POST请求时，它会验证Web服务器的响应。如果Web服务器返回200 OK状态码，那么证书申请就算完成了。如果Web服务器返回其他状态码，那么证书申请就失败了。http-01方式的缺点是Web服务器需要支持ACME协议，并且客户端必须信任那台Web服务器的可信任性。
- dns-01方式是利用域名服务器的DNS记录验证域名所有权。Let's Encrypt会向域名服务器的DNS记录中提交一个TXT记录，并等待验证。验证成功后，Let's Encrypt再向CA提交证书签名请求。dns-01方式的缺点是域名服务器必须支持DNSSEC，并且客户端必须信任那台域名服务器的可信任性。

# 4.核心算法原理和具体操作步骤以及数学公式讲解

## 域名验证过程

### 第一阶段：用户访问其域名对应的主页，然后浏览器向Let's Encrypt的ACME服务器请求证书


### 第二阶段：Let's Encrypt生成一个随机值，然后用这个随机值作为challenge并提交到域名服务器的DNS记录中。



```python
import time
import hashlib

def generate_challenge(domain):
    """Generate the challenge string"""
    nonce = ''.join([str(time.time()), str(random.randint(0, 9))])
    digest = hashlib.sha256(nonce.encode('utf-8')).hexdigest()[:10]
    return '_acme-challenge.' + domain + '.' + base64url_encode(digest).decode('ascii')
```

generate_challenge函数会生成challenge字符串。其中，base64url_encode函数用来将字节序列编码为ASCII字符串，保证可以用作DNS TXT记录的值。

注意：challenge字符串不是随机产生的，而是基于时间戳和随机数生成的，目的是避免在验证过程中篡改TXT记录。

### 第三阶段：当用户的域名服务器收到Let's Encrypt的提交后，会检测到TXT记录是否存在，如果存在则验证成功，否则返回错误信息。


### 第四阶段：如果域名服务器返回的TXT记录的内容与Let's Encrypt发送的challenge相符，那么就可以认为该域名属于当前用户，Let's Encrypt再向用户的服务器发出证书签名请求。


证书签名请求包括两部分：

1. 证书的请求数据（CSR文件）；
2. 用户私钥加密后的证书签名。

### 第五阶段：用户的服务器接收到证书签名请求后，开始生成私钥和CSR文件，并把它们提交给Let's Encrypt的CA。

### 第六阶段：CA生成一个证书文件，然后使用私钥加密证书，并且用Let's Encrypt的公钥签署证书。

### 第七阶段：Let's Encrypt服务器把证书发送回用户的服务器，然后保存并安装在服务器上。

### 操作步骤

1. 安装Let's Encrypt客户端工具，并执行“certbot --authenticator standalone”命令。
2. 执行“sudo certbot register –agree-tos –email <EMAIL>”命令，输入注册邮箱和用户名。
3. 执行“sudo certbot certonly -d example.com –manual”命令，输入域名example.com。
4. 在命令提示符下输入“y”，表示同意Let's Encrypt提供的CA机构颁发证书。
5. 执行“sudo mkdir /var/www/html/.well-known/acme-challenge/”命令，创建目录用于存放challenge。
6. 执行“echo “<random value>” > /var/www/html/.well-known/acme-challenge/<token>”命令，生成并写入challenge文件。
7. 通过浏览器访问http://example.com/.well-known/acme-challenge/<token>，查看challenge是否匹配。
8. 执行“sudo chmod 755 /var/www/html/.well-known/acme-challenge/”命令，授予challenge文件读写权限。
9. 执行“sudo chown root:root /var/www/html/.well-known/acme-challenge/”命令，赋予challenge文件属主和权限。
10. 在Let's Encrypt官网下载证书文件，并导入到您的服务器中。

## ACME协议原理

ACME协议是Let's Encrypt向CA机构发起申请证书的标准协议，分为注册阶段和认证阶段。

### 注册阶段

注册阶段是用户向证书颁发机构CA注册申请证书的前期准备，具体流程如下图所示：


### 认证阶段

认证阶段是用户向证书颁发机构CA提交认证申请的过程，具体流程如下图所示：


### Challenge类型

Challenge类型是ACME协议的一部分，用于确保域名所有权的有效性。一般来说，有两种Challenge类型：

1. HTTP Challenge：这种Challenge类型使用HTTP的方式向用户的服务器发起验证，需要用户服务器配置好ACME协议的响应接口。
2. DNS Challenge：这种Challenge类型使用DNS的方式向域名服务器的DNS记录发起验证，需要域名管理员提供相应的TXT记录。

### 更多信息


# 5.具体代码实例和解释说明

以下是具体的代码实例，供大家参考。

## Python客户端示例

```python
#!/usr/bin/env python
# coding=utf-8
from cryptography import x509
from acme import messages
from acme import challenges
from acme import client as acme_client
from acme import errors as acme_errors
from acme import jose
from OpenSSL import crypto
import logging
import sys

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class MyClient(acme_client.ClientV2):

    def __init__(self, key, directory, email):
        super().__init__(key=jose.JWKRSA(key),
                         net=None,
                         account=None,
                         verify_ssl=False,
                         log=LOGGER,
                         directory=directory)

        self._regr = None
        self._authzr = []
        self._challenges = {}
        self._responses = {}
        self._csr_pem = None

        # Register an account with the server using provided email address
        try:
            response = self.net.post(self.directory['newAccount'], json={
                'termsOfServiceAgreed': True,
                'contact': [
                    "mailto:{}".format(email)],
                'termsOfService': self.directory['meta']['termsOfService']])
            self._account = messages.NewRegistration.from_json(response.json())

            # Update contact information for existing accounts
            if not self._account.terms_of_service:
                LOGGER.warning("Updated your account to agree with the new terms of service")
            else:
                LOGGER.info("Terms of service already agreed.")

        except acme_errors.ConflictError:
            pass

    def _check_response(self, response):
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400 and \
                        response.headers.get('Content-Type', '').startswith(('application/problem+json',
                                                                             'application/json')):
            error = response.json()
            raise acme_errors.Error(error)
        else:
            raise acme_errors.unexpected_response(response)

    def request_issuance(self, csr_pem):
        # Generate a Certificate Signing Request (CSR) object from PEM encoded data
        self._csr_pem = csr_pem
        self._csr = x509.load_pem_x509_csr(csr_pem.encode(), backend=crypto.backend)

        # Prepare identifier authorization requests
        identifiers = [(id_obj.value, id_obj.value)
                       for id_obj in self._csr.subject.get_attributes_for_oid(x509.OID_COMMON_NAME)]
        authzrs = self.request_authorizations(identifiers)

        # Pick the first available authorization resource and fetch the associated challenges
        for authzr in authzrs:
            challbs = [challb for challb in authzr.body.challenges
                       if isinstance(challb.chall, challenges.HTTP01)]
            if challbs:
                challenge = challbs[0].chall
                break
        else:
            raise Exception("Unable to find supported challenge type")

        token = challenge.encode('token')
        url = challenge.path.lstrip('/')
        path = "{}/{}".format(self._regr.uri, url)
        content = "{token}.{thumbprint}".format(token=token, thumbprint=jose.b64(self._key.public_bytes()))
        response, validation = self.answer_challenge(challenge, content)

        self._responses[validation.body_hash.decode()] = {
            "resource": validation,
            "challenge": challenge}

        # Wait for the validation process to complete
        while self._responses.values():
            try:
                response = self.poll_and_request_issuance()

                if response is False:
                    continue

                validated_resources = set((res.identifier.value, res.body.uri) for res in response.body.validated_certificate_chain)
                outstanding_authorizations = [authzr for authzr in authzrs if any(ident.value == uri for ident, uri in validated_resources)]

                if not outstanding_authorizations:
                    chain_pems = [msg_as_pem(chain.body)
                                  for chain in sorted(response.body.certificates,
                                                    key=lambda c: int(c.url.rsplit('/', 1)[-1]))]

                    cert_pem = msg_as_pem(response.body.certificate)

                    bundle_pem = "".join(chain_pems[:-1]) + "\n" + cert_pem + "\n" + chain_pems[-1]

                    # Verify that the generated certificate can be loaded into an SSL context
                    ctx = ssl.create_default_context()
                    store = ctx.get_cert_store()
                    for pem in chain_pems:
                        store.add_cert(ssl.load_certificate(crypto.FILETYPE_PEM, pem))
                    ctx.load_verify_locations("/etc/ssl/certs/")
                    ctx.set_alpn_protocols(["h2", "http/1.1"])
                    conn = ctx.wrap_socket(socket.socket(),
                                            server_hostname="example.com",
                                            do_handshake_on_connect=True)
                    conn.connect(("example.com", 443))
                    conn.sendall("""GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n""".encode())
                    assert conn.recv(1024).find(b"\r\n\r\n<!DOCTYPE html>")!= -1
                    print(conn.version())
                    conn.close()

                    return {"certificate": bundle_pem}, ""

                for authzr in outstanding_authorizations:
                    self._authzr.remove(authzr)

            except KeyboardInterrupt:
                break

        return {}, "All attempts failed"

    def poll_and_request_issuance(self):
        # If we have no pending authorizations, exit early without making another request
        if not self._authzr or all(len(authzr.body.challenges) <= len(self._challenges[authzr])
                                    for authzr in self._authzr):
            return False

        responses = self.fetch_validations(*[(authzr, [_ch])
                                             for authzr in self._authzr
                                             for _ch in authzr.body.challenges
                                             if isinstance(_ch.chall, challenges.HTTP01)])

        updated_authzrs = set()
        resources = set()

        for response in responses:
            body = response.json()['body']
            key = body['keyAuthz']

            try:
                validation = self._responses[key]['resource']
            except KeyError:
                continue

            resource = messages.ValidationResource(
                validation.body.url, validation.body.body,
                encoding='base64', alg=jose.RS256)

            response = messages.StatusResponse(**self._check_response(self.net.post(body['uri'], headers={'Replay-Nonce': response.headers['Replay-Nonce']}, json={'status': 'valid'})))

            challenge = self._responses[key]['challenge']
            domain, validation_contents = b64_to_text(challenge.rsplit('.', 2)[1]).split('.')
            if sha256(validation_contents) == validation.body_hash.decode():
                self._responses[key]["validated"] = True
                updated_authzrs.add(self._responses[key]['authorization'])
                resources.add(resource)

        self._authzr = list(updated_authzrs)

        if resources:
            return self.request_issuance({})

        return False


def main():
    directory = acme_client.Directory.from_json(requests.get("https://acme-v02.api.letsencrypt.org/directory").json())

    # Load or create private RSA key
    key = jose.JWKRSA.load(open('/etc/pki/cert/private/myserver.key', 'rb').read())

    # Create the client instance
    client = MyClient(key, directory, "<EMAIL>")

    # Read the CSR file
    with open('/tmp/myserver.csr', 'rt') as f:
        csr_pem = f.read()

    # Request issuance of a new certificate
    result, message = client.request_issuance(csr_pem)

    # Write out resulting certificates and chain
    if result:
        bundle_pem = result["certificate"]
        with open('/etc/httpd/ssl/myserver.crt', 'wt') as f:
            f.write(bundle_pem)
            os.chmod('/etc/httpd/ssl/myserver.crt', stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        with open('/etc/httpd/ssl/myserver.ca.crt', 'wt') as f:
            f.write("\n".join(filter(bool, re.findall(r'-----BEGIN CERTIFICATE-----([^-\n]+)\n.*?\n.*?\n.*?\n------END CERTIFICATE-----', bundle_pem))))
            os.chmod('/etc/httpd/ssl/myserver.ca.crt', stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        with open('/etc/httpd/ssl/myserver.key', 'wb') as f:
            f.write(key.export_private())
            os.chmod('/etc/httpd/ssl/myserver.key', stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        print("Issued new certificate!")
    else:
        print("Failed to issue new certificate: {}".format(message))


if __name__ == '__main__':
    sys.exit(main())
```

## Go客户端示例

```go
package main

import (
	"fmt"
	"os"

	"github.com/xenolf/lego/acme"
)

func main() {
	// Prepare the required input files and directories
	dirURL := "https://acme-staging-v02.api.letsencrypt.org/directory"
	privKeyFile := "./privkey.pem"
	pubKeyFile := "./pubkey.pem"
	csrFile := "./csr.pem"
	certFile := "/var/lib/apache2/ssl/certificate.pem"
	chainFile := "/var/lib/apache2/ssl/intermediate.pem"

	// Check if privkey exists and read it
	_, err := os.Stat(privKeyFile)
	if err == nil {
		key, err := ioutil.ReadFile(privKeyFile)
		if err!= nil {
			panic(err)
		}

		fmt.Println("Using existing private key...")
		client := &acme.Client{Key: *jose.ParseJWK(string(key))}
	} else {
		fmt.Println("Generating a new private key...")
		key, err := rsa.GenerateKey(rand.Reader, 4096)
		if err!= nil {
			panic(err)
		}

		if _, err := os.Stat("./"); os.IsNotExist(err) {
			err := os.MkdirAll("./", 0755)
			if err!= nil {
				panic(err)
			}

			defer os.Remove(privKeyFile)
			defer os.Remove(pubKeyFile)
		}

		publicKey, privateKey := jose.SSHAlgorithm{}.Wrap(key)
		ioutil.WriteFile(privKeyFile, []byte(privateKey), 0600)
		ioutil.WriteFile(pubKeyFile, []byte(publicKey), 0644)
		fmt.Printf("Private key saved at %q...\n", privKeyFile)

		client := &acme.Client{Key: *jose.JSONWebKeyFromBytes([]byte(publicKey)), DirectoryURL: dirURL}
	}

	// Read the CSR file
	fmt.Println("Reading CSR...")
	csrData, err := ioutil.ReadFile(csrFile)
	if err!= nil {
		panic(err)
	}

	// Process the CSR using the ACME library
	order, err := client.CreateOrder(csrData)
	if err!= nil {
		panic(err)
	}

	fmt.Printf("Waiting for order %q to complete...\n", order.URI)
	for!order.IsFinal() {
		time.Sleep(5 * time.Second)
		order, err = client.GetOrder(order.URI)
		if err!= nil {
			panic(err)
		}
	}

	fmt.Printf("Order status: %s\n", order.Status)
	if order.Status == acme.Invalid || order.Status == acme.Expired {
		fmt.Println("Order failed! Deleting private key...")
		os.Remove(privKeyFile)
		os.Remove(pubKeyFile)
		return
	}

	// Retrieve the finalized certificate and intermediate certificates
	certs := [][]byte{}
	for _, authz := range order.Authorizations {
		authzRes, err := client.GetAuthorization(authz)
		if err!= nil {
			panic(err)
		}

		certsData, err := getCertificate(client, authzRes)
		if err!= nil {
			panic(err)
		}

		certs = append(certs, certsData...)
	}

	fmt.Printf("%d certificate(s) issued:\n%s\n", len(certs)-1, strings.Join(strings.Split(string(certs[0]), "\n")[1:], "")) // remove header line

	// Save the signed certificate, including its chain of trust
	signedCert := string(certs[0])
	intermediateCerts := string(certs[1:])
	saveToFile(certFile, signedCert+"\n"+intermediateCerts)
	os.Chmod(certFile, 0644)

	// Extract the intermediate certificates from the signed certificate
	intermediateCertsOnly := extractIntermediateCerts(signedCert, intermediateCerts)
	saveToFile(chainFile, intermediateCertsOnly)
	os.Chmod(chainFile, 0644)

	fmt.Println("Certificate and chain saved successfully!")
}

func saveToFile(filename, contents string) {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err!= nil {
		panic(err)
	}
	defer f.Close()

	_, err = f.WriteString(contents)
	if err!= nil {
		panic(err)
	}
}

func getCertificate(client *acme.Client, authzRes *acme.Authorization) ([]byte, error) {
	for _, chall := range authzRes.Challenges {
		switch chall.Type {
		case "tls-sni-01":
			resp, err := client.AcceptTLSSNI01Challenge(chall.Token)
			if err!= nil {
				return nil, err
			}

			if resp.Status == "invalid" {
				continue
			}

			fmt.Println("Successfully verified tls-sni-01 challenge")
		case "http-01":
			resp, err := client.AcceptHTTP01Challenge(chall.URL)
			if err!= nil {
				return nil, err
			}

			fmt.Printf("Successfully obtained http-01 challenge response for %s.\n", resp.URI)

			waitTime := 2
			maxWait := waitTime * 20

			client.HTTPClient = &http.Client{Timeout: time.Duration(waitTime * time.Second)}

			for i := 0; i*waitTime < maxWait &&!isValidHttp01Response(resp); i++ {
				time.Sleep(waitTime * time.Second)

				resp, err = client.GetChallenge(resp.URI)
				if err!= nil {
					return nil, fmt.Errorf("failed to check challenge: %s", err)
				}

				fmt.Printf("Checking http-01 challenge status (%d/%d)...\n", i+1, maxWait/waitTime)
			}

			if i*waitTime >= maxWait {
				return nil, fmt.Errorf("timed out waiting for http-01 verification (%ds)", maxWait)
			}

			fmt.Println("Successfully verified http-01 challenge")
		default:
			return nil, fmt.Errorf("unknown challenge type: %s", chall.Type)
		}
	}

	return client.GetCertificate(authzRes.Certificates...), nil
}

func isValidHttp01Response(resp *acme.Challenge) bool {
	content, err := getContent(resp.URI)
	if err!= nil {
		return false
	}

	return sha256([]byte(content)) == resp.KeyAuthorization
}

func getContent(url string) (string, error) {
	resp, err := http.DefaultClient.Head(url)
	if err!= nil {
		return "", err
	}

	if resp.StatusCode!= http.StatusOK {
		return "", fmt.Errorf("received non-OK response code: %d", resp.StatusCode)
	}

	locationHeader := resp.Header.Get("Location")
	if locationHeader == "" {
		return "", fmt.Errorf("missing Location header on successful GET")
	}

	return getContent(locationHeader)
}

func extractIntermediateCerts(certPem, intermediatesPem string) string {
	block, rest := pem.Decode([]byte(certPem))
	if block == nil || block.Type!= "CERTIFICATE" {
		return ""
	}

	intermediates := []*x509.Certificate{}
	for len(rest) > 0 {
		innerBlock, innerRest := pem.Decode(rest)
		if innerBlock == nil || innerBlock.Type!= "CERTIFICATE" {
			break
		}

		intermediates = append(intermediates, parseCertificate(innerBlock.Bytes))
		rest = innerRest
	}

	intPEMBlocks := make([][]byte, len(intermediates)+1)
	intPEMBlocks[0] = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: block.Bytes})
	for i, intermediate := range intermediates {
		intPEMBlocks[i+1] = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: intermediate.Raw})
	}

	return string(bytes.Join(intPEMBlocks, []byte("\n")))
}

func parseCertificate(data []byte) *x509.Certificate {
	cert, err := x509.ParseCertificate(data)
	if err!= nil {
		panic(err)
	}

	return cert
}
```