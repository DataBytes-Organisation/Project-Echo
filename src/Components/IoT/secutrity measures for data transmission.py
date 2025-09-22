#!/usr/bin/env python3
"""
Generates a Root CA, a broker/server cert, and N device certs (ECDSA P-256).
Outputs PEMs + per-device provisioning bundles for mTLS (MQTT/CoAPS).
Usage:
  python pki/provision.py --out docker/mosquitto/certs --num-devices 3 --cn-prefix iotnode-
Requires:
  pip install cryptography
"""
import os, argparse, secrets, json, datetime
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

def write_pem(path, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f: f.write(data)

def mk_name(cn:str):
    return x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EchoNet SecLab"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])

def mk_cert(subject, issuer, pubkey, issuer_key, is_ca=False, san_dns=None, eku_server=False, eku_client=False, days=3650):
    now = datetime.datetime.utcnow()
    builder = (x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(pubkey)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(pubkey), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False))
    eku = []
    if eku_server: eku.append(ExtendedKeyUsageOID.SERVER_AUTH)
    if eku_client: eku.append(ExtendedKeyUsageOID.CLIENT_AUTH)
    if eku:
        builder = builder.add_extension(x509.ExtendedKeyUsage(eku), critical=False)
    if san_dns:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(san_dns)]), critical=False)
    return builder.sign(private_key=issuer_key, algorithm=hashes.SHA256())

def pem_privkey(priv):
    return priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption())

def pem_cert(cert):
    return cert.public_bytes(serialization.Encoding.PEM)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-devices", type=int, default=1)
    ap.add_argument("--broker-cn", default="mqtt-broker.local")
    ap.add_argument("--cn-prefix", default="device-")
    args = ap.parse_args()

    out = args.out
    # Root CA
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = mk_cert(
        subject=mk_name("EchoNet Root CA"),
        issuer=mk_name("EchoNet Root CA"),
        pubkey=ca_key.public_key(),
        issuer_key=ca_key,
        is_ca=True, days=3650)
    write_pem(f"{out}/rootCA.key.pem", pem_privkey(ca_key))
    write_pem(f"{out}/rootCA.cert.pem", pem_cert(ca_cert))

    # Broker/server
    broker_key = ec.generate_private_key(ec.SECP256R1())
    broker_cert = mk_cert(
        subject=mk_name(args.broker_cn),
        issuer=ca_cert.subject,
        pubkey=broker_key.public_key(),
        issuer_key=ca_key,
        san_dns=args.broker_cn, eku_server=True, days=825)
    write_pem(f"{out}/broker.key.pem", pem_privkey(broker_key))
    write_pem(f"{out}/broker.cert.pem", pem_cert(broker_cert))

    # Devices
    for i in range(args.num_devices):
        cn = f"{args.cn_prefix}{i+1}"
        d_key = ec.generate_private_key(ec.SECP256R1())
        d_cert = mk_cert(
            subject=mk_name(cn),
            issuer=ca_cert.subject,
            pubkey=d_key.public_key(),
            issuer_key=ca_key,
            eku_client=True, days=825)
        dev_dir = f"{out}/devices/{cn}"
        write_pem(f"{dev_dir}/{cn}.key.pem", pem_privkey(d_key))
        write_pem(f"{dev_dir}/{cn}.cert.pem", pem_cert(d_cert))
        write_pem(f"{dev_dir}/rootCA.cert.pem", pem_cert(ca_cert))
        # Provisioning bundle (for device filesystem)
        bundle = {
            "device_id": cn,
            "mqtt": {
                "host": "localhost",
                "port_tls": 8883,
                "topic": f"d/{cn}/telemetry",
                "cafile": "rootCA.cert.pem",
                "certfile": f"{cn}.cert.pem",
                "keyfile": f"{cn}.key.pem"
            },
            "coaps": {
                "host": "localhost",
                "port": 5684,
                "psk_hint": "echo-psk",  # for PSK option
                "psk": secrets.token_hex(16)
            }
        }
        os.makedirs(dev_dir, exist_ok=True)
        with open(f"{dev_dir}/provisioning.json", "w") as f:
            json.dump(bundle, f, indent=2)
    print(f"Provisioned -> {out}")

if __name__ == "__main__":
    main()
