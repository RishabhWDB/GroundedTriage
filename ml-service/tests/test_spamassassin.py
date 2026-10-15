import io
import tarfile

from data.spamassassin import build_frame, iter_messages, parse_message

PLAIN = (
    b"From: winner@prizes.biz\r\nSubject: You have WON $1,000,000!!!\r\n"
    b"Content-Type: text/plain; charset=us-ascii\r\n\r\n"
    b"Congratulations! Click the link below to claim your prize money today, act now.\r\n"
)
HTML = (
    b"From: deals@cheap.biz\r\nSubject: =?utf-8?q?Cheap_meds?=\r\n"
    b"Content-Type: text/html; charset=iso-8859-1\r\n\r\n"
    b"<html><body><p>Buy cheap medication online with no prescription needed.</p>"
    b"<p>Best prices guaranteed &amp; fast shipping!</p></body></html>\r\n"
)
SPANISH = (
    b"Subject: Oferta especial\r\nContent-Type: text/plain\r\n\r\n"
    b"Compre ahora los mejores productos con descuento, "
    b"oferta valida solo por hoy en nuestra tienda.\r\n"
)
SHORT = b"Subject: hi\r\n\r\nbuy now\r\n"


def test_parse_message_plain_and_html():
    subj, body = parse_message(PLAIN)  # raw content; clean_body normalises line endings later
    assert subj == "You have WON $1,000,000!!!"
    assert (
        body.strip()
        == "Congratulations! Click the link below to claim your prize money today, act now."
    )
    subj, body = parse_message(HTML)
    assert subj == "Cheap meds" and "<p>" in body  # HTML is converted by clean_body later


def test_build_frame_filters_and_dedupes():
    msgs = [
        ("a", PLAIN),
        ("b", HTML),
        ("c", SPANISH),
        ("d", SHORT),
        ("e", PLAIN),
        ("f", b"\xff\xfe garbage"),
    ]
    df, st = build_frame(msgs)
    assert st.non_english == 1 and st.too_short >= 1 and st.duplicates_removed == 1
    assert len(df) == 2 and set(df.category) == {"Spam"} and set(df.origin) == {"real"}
    assert df.urgency.isna().all()
    assert (
        "<p>" not in df.body.iloc[1]
        and "Best prices guaranteed & fast shipping!" in df.body.iloc[1]
    )
    assert df.id.is_unique


def test_iter_messages_reads_without_extracting(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in [("spam/0001.abc", PLAIN), ("spam/cmds", b"rm -rf /")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path = tmp_path / "x.tar.bz2"
    path.write_bytes(buf.getvalue())
    got = list(iter_messages(path))
    assert got == [("spam/0001.abc", PLAIN)]
    assert not (tmp_path / "spam").exists()
