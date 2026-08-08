/**
 * @jest-environment jsdom
 *
 * needs a real DOM here because showToast touches document.body/document.head -
 * everywhere else we test we just stub showToast out instead, but here it's the
 * thing actually being exercised (indirectly, via withRetry)
 */

const { getApiErrorMessage, withRetry, showToast } = require('../HMI-utils.js');

describe('getApiErrorMessage - safe user-facing messages', () => {
  it('gives a clear message for a timed-out request', () => {
    const msg = getApiErrorMessage({ code: 'ECONNABORTED' });
    expect(msg).toMatch(/timed out/i);
  });

  it('gives an authorisation message for 401/403, without leaking backend detail', () => {
    const msg = getApiErrorMessage({ response: { status: 401 } });
    expect(msg).toMatch(/not authorised/i);
  });

  it('gives a not-found message for 404', () => {
    const msg = getApiErrorMessage({ response: { status: 404 } });
    expect(msg).toMatch(/not found/i);
  });

  it('gives a generic server-unavailable message for 5xx, not the raw stack trace', () => {
    const msg = getApiErrorMessage({ response: { status: 503 } });
    expect(msg).toMatch(/server is currently unavailable/i);
  });

  it('falls back to the error message for anything else', () => {
    const msg = getApiErrorMessage(new Error('some specific problem'));
    expect(msg).toBe('some specific problem');
  });

  it('falls back to the default fallback message when there is nothing to go on', () => {
    const msg = getApiErrorMessage(null);
    expect(msg).toBe('Something went wrong. Please try again.');
  });
});

describe('withRetry', () => {
  it('retries the given number of times and eventually throws if it never succeeds', async () => {
    const alwaysFails = jest.fn().mockRejectedValue(new Error('nope'));

    await expect(
      withRetry(alwaysFails, { attempts: 3, delayMs: 1 })
    ).rejects.toThrow('nope');

    expect(alwaysFails).toHaveBeenCalledTimes(3);
  });

  it('stops retrying as soon as it succeeds', async () => {
    const failsOnceThenWorks = jest
      .fn()
      .mockRejectedValueOnce(new Error('one blip'))
      .mockResolvedValueOnce('all good');

    const result = await withRetry(failsOnceThenWorks, { attempts: 3, delayMs: 1 });

    expect(result).toBe('all good');
    expect(failsOnceThenWorks).toHaveBeenCalledTimes(2);
  });

  // toasts from earlier tests linger in the shared jsdom document (they only
  // self-remove on a transitionend that never fires here), so count the delta
  // rather than the absolute number on the page
  const countToasts = () => document.querySelectorAll('.hmi-toast').length;

  it('shows a retry toast per failed attempt by default', async () => {
    const before = countToasts();
    const alwaysFails = jest.fn().mockRejectedValue(new Error('nope'));

    await expect(
      withRetry(alwaysFails, { attempts: 3, delayMs: 1 })
    ).rejects.toThrow('nope');

    // one toast between each pair of attempts, so attempts - 1
    expect(countToasts() - before).toBe(2);
  });

  it('shows no toasts at all when silent, but still retries the same number of times', async () => {
    const before = countToasts();
    const alwaysFails = jest.fn().mockRejectedValue(new Error('nope'));

    await expect(
      withRetry(alwaysFails, { attempts: 3, delayMs: 1, silent: true })
    ).rejects.toThrow('nope');

    expect(alwaysFails).toHaveBeenCalledTimes(3);
    expect(countToasts() - before).toBe(0);
  });
});
