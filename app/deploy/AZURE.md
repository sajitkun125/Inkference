# Deploying Inkference to Azure Container Apps

Accounts run on PostgreSQL and sign-in can go through Google or Microsoft Entra ID.
This is the end-to-end setup, in the order the pieces depend on each other.

Nothing here is secret-bearing — every credential is referenced, never written down.

---

## 1. What you need first

| Thing | Why |
|---|---|
| Azure subscription + `az` CLI (`az login`) | everything below |
| Azure Container Registry, or any registry the app can pull from | the image |
| Azure Database for PostgreSQL **Flexible Server** | accounts, sessions, identities |
| A Google Cloud project *(optional)* | "Continue with Google" |
| An Entra ID tenant *(optional)* | "Microsoft" / institutional SSO |

Set these once so the commands below are copy-pasteable:

```bash
RG=inkference-rg
LOC=westeurope
ACR=inkferenceacr           # must be globally unique
APP=inkference
ENVNAME=inkference-env
PG=inkference-pg            # must be globally unique
```

---

## 2. PostgreSQL

```bash
az postgres flexible-server create \
  --resource-group $RG --name $PG --location $LOC \
  --tier Burstable --sku-name Standard_B1ms \
  --version 16 --database-name inkference \
  --admin-user inkferenceadmin --admin-password '<a-strong-password>' \
  --public-access 0.0.0.0            # Azure services only; see the note below
```

Two things worth getting right now rather than later:

- **TLS is mandatory.** The connection string must end in `?sslmode=require`. Without
  it the driver connects in plaintext and the server closes the socket, which surfaces
  as an unhelpful "connection unexpectedly closed" at startup.
- **`--public-access 0.0.0.0` allows all Azure IPs, not the whole internet** — but it
  is still broader than it needs to be. For anything real, put the server on a VNet
  and give the Container App a private endpoint instead.

`B1ms` caps out around 50 connections. `DB_POOL_SIZE` (5) + `DB_MAX_OVERFLOW` (5) per
replica means ten replicas would exhaust it, so lower the pool or raise the tier before
scaling that far.

---

## 3. Google sign-in

[Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services**:

1. **OAuth consent screen** → External (or Internal for a Workspace-only app). Fill in
   the app name, support email, and the `openid`, `email`, `profile` scopes. Nothing
   here needs Google's review — those three are non-sensitive.
2. **Credentials → Create credentials → OAuth client ID → Web application.**
3. **Authorized redirect URIs** — add exactly:

   ```
   https://<your-app-fqdn>/api/auth/oidc/google/callback
   ```

   Byte-for-byte. A trailing slash, `http` instead of `https`, or the wrong subdomain
   all produce `redirect_uri_mismatch` and nothing else. You will not know the FQDN
   until step 5, so either come back here afterwards or pre-assign a custom domain.

4. Keep the **client ID** and **client secret**.

For local development add a second redirect URI:
`http://localhost:8000/api/auth/oidc/google/callback`.

---

## 4. Microsoft Entra ID sign-in

[Entra admin center](https://entra.microsoft.com/) → **App registrations → New registration**:

1. **Supported account types** decides who can sign in, and it must agree with
   `MICROSOFT_OAUTH_TENANT`:

   | Registration setting | `MICROSOFT_OAUTH_TENANT` |
   |---|---|
   | This organizational directory only | your tenant GUID |
   | Any organizational directory | `organizations` *(the default)* |
   | …and personal Microsoft accounts | `common` |

   The app defaults to `organizations` deliberately. Under `common`, personal accounts
   can sign in, and their email addresses are self-asserted — the app refuses to link
   those to an existing account (see "Account linking" below), so users hit a confusing
   dead end. Prefer a tenant GUID or `organizations`.

2. **Redirect URI** → platform **Web** →
   `https://<your-app-fqdn>/api/auth/oidc/microsoft/callback`
3. **Certificates & secrets → New client secret.** Copy the *Value*, not the Secret ID.
   Note the expiry — Entra caps secrets at 24 months and sign-in breaks the day it lapses.
4. **API permissions** → `openid`, `email`, `profile` (Microsoft Graph, delegated).
   These are default-consented; no admin approval needed.

---

## 5. Build, push, deploy

```bash
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
az acr build -r $ACR -t inkference:latest -f app/deploy/Dockerfile.azure .

az containerapp env create -g $RG -n $ENVNAME -l $LOC

az containerapp create \
  -g $RG -n $APP --environment $ENVNAME \
  --image $ACR.azurecr.io/inkference:latest \
  --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 3 \
  --cpu 2 --memory 4Gi
```

TrOCR and the embedding model are what set the memory floor — 4 GiB is the practical
minimum, and the first request after a cold start pays for loading them. `--min-replicas 1`
rather than 0 for that reason: scale-to-zero would put a model load in front of a
visitor every time the app went quiet.

Read the FQDN, then go back and register the two redirect URIs from steps 3 and 4:

```bash
az containerapp show -g $RG -n $APP --query properties.configuration.ingress.fqdn -o tsv
```

---

## 6. Secrets and configuration

```bash
az containerapp secret set -g $RG -n $APP --secrets \
  database-url="postgresql://inkferenceadmin:<pw>@$PG.postgres.database.azure.com:5432/inkference?sslmode=require" \
  secret-key="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  google-client-id="<...>" \
  google-client-secret="<...>" \
  ms-client-id="<...>" \
  ms-client-secret="<...>" \
  groq-api-key="<...>" \
  gemini-api-key="<...>"

az containerapp update -g $RG -n $APP --set-env-vars \
  DATABASE_URL=secretref:database-url \
  INKFERENCE_SECRET_KEY=secretref:secret-key \
  GOOGLE_OAUTH_CLIENT_ID=secretref:google-client-id \
  GOOGLE_OAUTH_CLIENT_SECRET=secretref:google-client-secret \
  MICROSOFT_OAUTH_CLIENT_ID=secretref:ms-client-id \
  MICROSOFT_OAUTH_CLIENT_SECRET=secretref:ms-client-secret \
  MICROSOFT_OAUTH_TENANT=organizations \
  GROQ_API_KEY=secretref:groq-api-key \
  GEMINI_API_KEY=secretref:gemini-api-key \
  INKFERENCE_PUBLIC_BASE_URL="https://$(az containerapp show -g $RG -n $APP --query properties.configuration.ingress.fqdn -o tsv)"
```

**`INKFERENCE_SECRET_KEY` must be identical on every replica.** It signs the short-lived
cookie that carries a sign-in between the redirect out to the provider and the callback
back. A sign-in that starts on replica A and returns to replica B fails outright if the
two disagree, and the app falls back to a per-process random key when the variable is
unset — which works on one replica and breaks silently on two.

**`INKFERENCE_PUBLIC_BASE_URL`** is what makes the OAuth redirect URI come out as
`https://<fqdn>/...` rather than the internal address the container actually sees. The
app can derive it from `X-Forwarded-*` too, but setting it explicitly removes a whole
class of "worked locally, `redirect_uri_mismatch` in Azure".

For anything beyond a first deploy, move these to **Key Vault** and give the Container
App a managed identity, so rotation does not mean re-running this command.

---

## 7. Health probes

```bash
az containerapp update -g $RG -n $APP \
  --liveness-probe-path /api/health   --liveness-probe-initial-delay 60 \
  --readiness-probe-path /api/ready
```

The split matters. `/api/health` answers from process state alone, so a database blip
cannot get healthy containers restarted. `/api/ready` checks PostgreSQL and returns
503 when it is unreachable, which pulls that replica out of the ingress rotation
instead of letting it answer every sign-in with a 500. It also reports the applied
schema revision, which is the quickest way to spot a half-finished rollout.

---

## 8. Migrations

The app runs `alembic upgrade head` at startup, serialised across replicas by a
PostgreSQL advisory lock — the first replica migrates, the rest wait and then find
nothing to do.

That is fine for this schema. Once a migration takes long enough to matter, move it out:

```bash
az containerapp update -g $RG -n $APP --set-env-vars DB_MIGRATE_ON_STARTUP=false
az containerapp job create ...   # run `alembic upgrade head` as a pre-deploy job
```

Otherwise a slow migration holds every replica's startup open behind the lock.

---

## Account linking, and why an unverified email is refused

One person can hold several ways in — a password, Google, Entra — all pointing at one
account and one library. The join is the **verified** email address:

- Signed up with a password, later clicks *Continue with Google* → same account.
- Signs in with Google, then Entra, same address → same account.
- Provider reports the address as unverified → **refused outright.**

That last rule is the load-bearing one. Because a provider sign-in adopts an existing
account with a matching address, accepting an unverified claim would let anyone able to
make a provider emit `someone@example.com` walk into that person's library. Google
reports `email_verified`; Entra omits it, so a work/school account is trusted (the
tenant administers the domain) and a personal Microsoft account is not (its holder
types their own address).

Identities are keyed on the provider's immutable subject id, not the address, so a user
who renames their mailbox keeps their library.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `redirect_uri_mismatch` | The registered URI differs from `INKFERENCE_PUBLIC_BASE_URL` + `/api/auth/oidc/<provider>/callback`. Compare them character by character, including the scheme and any trailing slash. |
| Sign-in works, then "state mismatch" | `INKFERENCE_SECRET_KEY` unset or differing across replicas. |
| Buttons render greyed out | Client id or secret missing. `GET /api/auth/providers` reports what the backend actually sees. |
| Container never becomes healthy | `DATABASE_URL` wrong or unreachable. The startup log names the host with the password blanked; check `?sslmode=require`. |
| `AADSTS7000215` | Entra client secret is wrong — the *Value* was needed, not the Secret ID. |
| `AADSTS50011` | Redirect URI not registered on the Entra app. |
| Sessions vanish on deploy | `DATABASE_URL` points somewhere ephemeral. Sessions live in Postgres precisely so they survive. |
| First request after idle 500s | Pool holding connections Azure's gateway already dropped. `DB_POOL_RECYCLE` (180s) should sit under the gateway's ~4 minute idle timeout. |
