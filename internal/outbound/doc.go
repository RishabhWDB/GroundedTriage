// Package outbound sends approved replies through the originating source
// adapter (P6.3): exactly one send per approval with respect to this
// system's state, and never a blind resend after a crash mid-send.
package outbound
