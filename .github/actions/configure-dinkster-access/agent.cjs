const { execFileSync } = require('node:child_process');
const { appendFileSync, writeFileSync } = require('node:fs');
const { join } = require('node:path');

const privateKey = process.env['INPUT_DEPLOY-KEY'];
delete process.env['INPUT_DEPLOY-KEY'];
const bin = process.platform === 'win32'
  ? `${process.env.ProgramFiles}/Git/usr/bin/`
  : '';
const agent = `${bin}ssh-agent${process.platform === 'win32' ? '.exe' : ''}`;

try {
  if (process.env.STATE_DINKSTER_POST) {
    if (!process.env.STATE_DINKSTER_AGENT_PID) return;
    execFileSync(agent, ['-k'], {
      env: {
        ...process.env,
        SSH_AGENT_PID: process.env.STATE_DINKSTER_AGENT_PID,
        SSH_AUTH_SOCK: process.env.STATE_DINKSTER_AUTH_SOCK,
      },
      stdio: 'pipe',
    });
  } else {
    appendFileSync(process.env.GITHUB_STATE, 'DINKSTER_POST=true\n');
    if (!privateKey?.trim()) throw new Error('Missing Dinkster deploy key');
    const output = execFileSync(agent, ['-s'], { encoding: 'utf8' });
    const pid = output.match(/^SSH_AGENT_PID=(\d+);/m)?.[1];
    const socket = output.match(/^SSH_AUTH_SOCK=([^;]+);/m)?.[1];
    if (!pid || !socket) throw new Error('Invalid SSH agent response');
    appendFileSync(process.env.GITHUB_STATE,
      `DINKSTER_AGENT_PID=${pid}\nDINKSTER_AUTH_SOCK=${socket}\n`);
    execFileSync(`${bin}ssh-add${process.platform === 'win32' ? '.exe' : ''}`, ['-'], {
      input: `${privateKey.trim()}\n`,
      env: { ...process.env, SSH_AGENT_PID: pid, SSH_AUTH_SOCK: socket },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const knownHosts = join(process.env.RUNNER_TEMP, 'dinkster-known-hosts');
    writeFileSync(knownHosts,
      'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n',
      { mode: 0o600 });
    const ssh = `${bin}ssh${process.platform === 'win32' ? '.exe' : ''}`;
    const sshCommand = `"${ssh.replaceAll('\\', '/')}" -o StrictHostKeyChecking=yes -o UserKnownHostsFile="${knownHosts.replaceAll('\\', '/')}"`;
    appendFileSync(process.env.GITHUB_ENV, [
      `SSH_AGENT_PID=${pid}`,
      `SSH_AUTH_SOCK=${socket}`,
      'GIT_CONFIG_COUNT=2',
      'GIT_CONFIG_KEY_0=url.ssh://git@github.com/Kosinkadink/Dinkster.insteadOf',
      'GIT_CONFIG_VALUE_0=https://github.com/Kosinkadink/Dinkster',
      'GIT_CONFIG_KEY_1=core.sshCommand',
      `GIT_CONFIG_VALUE_1=${sshCommand}`,
      '',
    ].join('\n'));
  }
} catch {
  console.error('::error::Dinkster SSH agent setup or cleanup failed. Check the deploy key and Git SSH installation.');
  process.exitCode = 1;
}
