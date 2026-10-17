-- +goose Up
-- P3.10: the binary spam filter runs before the topic classifier. A ticket it
-- mutes is stored with tier 'spam_filter'; every classification keeps the
-- filter's score and whether it fell in the "possible spam" review band.
alter table classifications drop constraint if exists classifications_tier_check;
alter table classifications
    add constraint classifications_tier_check check (tier in ('fast', 'deep', 'spam_filter'));
alter table classifications add column if not exists spam_score real
    check (spam_score is null or (spam_score >= 0 and spam_score <= 1));
alter table classifications add column if not exists possible_spam boolean not null default false;

-- +goose Down
alter table classifications drop column if exists possible_spam;
alter table classifications drop column if exists spam_score;
alter table classifications drop constraint if exists classifications_tier_check;
alter table classifications add constraint classifications_tier_check check (tier in ('fast', 'deep'));
