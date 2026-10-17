package scheduler

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"golang.org/x/oauth2"
)

// RunInstalledAppFlow runs the OAuth2 installed-app (loopback) flow: it
// listens on 127.0.0.1:<random port>, shows the consent URL (via show),
// waits for Google's redirect, checks the state, exchanges the code with
// PKCE and saves the token to tokenFile (mode 0600).
func RunInstalledAppFlow(ctx context.Context, cfg *oauth2.Config, tokenFile string, show func(url string)) (*oauth2.Token, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	defer ln.Close()
	c := *cfg
	c.RedirectURL = fmt.Sprintf("http://%s/callback", ln.Addr())
	state, err := randomState()
	if err != nil {
		return nil, err
	}
	verifier := oauth2.GenerateVerifier()
	show(c.AuthCodeURL(state, oauth2.AccessTypeOffline, oauth2.ApprovalForce, oauth2.S256ChallengeOption(verifier)))

	codes := make(chan string, 1)
	errs := make(chan error, 1)
	srv := &http.Server{Handler: CallbackHandler(state, codes, errs), ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	var code string
	select {
	case code = <-codes:
	case err := <-errs:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	tok, err := c.Exchange(ctx, code, oauth2.VerifierOption(verifier))
	if err != nil {
		return nil, fmt.Errorf("token exchange: %w", err)
	}
	if err := SaveToken(tokenFile, tok); err != nil {
		return nil, err
	}
	return tok, nil
}

// CallbackHandler receives the OAuth redirect: a matching state yields the
// code; a wrong state or an error parameter is rejected.
func CallbackHandler(state string, codes chan<- string, errs chan<- error) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/callback" {
			http.NotFound(w, r)
			return
		}
		q := r.URL.Query()
		switch {
		case q.Get("error") != "":
			http.Error(w, "authorisation failed: "+q.Get("error"), http.StatusBadRequest)
			trySend(errs, fmt.Errorf("authorisation failed: %s", q.Get("error")))
		case q.Get("state") != state:
			http.Error(w, "state mismatch", http.StatusBadRequest)
			trySend(errs, errors.New("state mismatch (possible CSRF); start again"))
		case q.Get("code") == "":
			http.Error(w, "missing code", http.StatusBadRequest)
		default:
			_, _ = w.Write([]byte("GroundedTriage is authorised for Google Calendar. You can close this tab."))
			select {
			case codes <- q.Get("code"):
			default:
			}
		}
	})
}

func trySend(ch chan<- error, err error) {
	select {
	case ch <- err:
	default:
	}
}

func randomState() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
