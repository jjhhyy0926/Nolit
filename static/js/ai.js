/* AI 추천 페이지 — 채팅 로직 */
let currentStep = 0;

const $chatWrap   = document.getElementById('chat-wrap');
const $typing     = document.getElementById('typing');
const $input      = document.getElementById('chat-input');
const $sendBtn    = document.getElementById('send-btn');
const $recSection = document.getElementById('rec-section');
const $recList    = document.getElementById('rec-list');

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

function scrollChat() {
    $chatWrap.scrollTop = $chatWrap.scrollHeight;
}

function escHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');
}

function typeMessage(element, text, speed = 25) {
    return new Promise((resolve) => {
        const formatted = text.replace(/\s*·/g, '\n·');
        element.style.whiteSpace = 'pre-wrap';
        element.textContent = '';
        let i = 0;
        const timer = setInterval(() => {
            element.textContent += formatted[i];
            i++;
            scrollChat();
            if (i >= formatted.length) {
                clearInterval(timer);
                resolve();
            }
        }, speed);
    });
}

// 첫 AI 메시지
typeMessage(document.getElementById('first-msg'), AI_FLOW[0]);

async function appendMsg(role, text) {
    const row = document.createElement('div');
    if (role === 'user') {
        row.className = 'chat-row-user';
        row.innerHTML = `<div class="chat-user">${escHtml(text)}</div>`;
        $chatWrap.insertBefore(row, $typing);
        scrollChat();
    } else {
        row.className = 'chat-row-ai';
        row.innerHTML = `
            <div class="chat-avatar">✦</div>
            <div>
                <div class="chat-label">NOLIT AI</div>
                <div class="chat-ai"></div>
            </div>`;
        $chatWrap.insertBefore(row, $typing);
        scrollChat();
        const target = row.querySelector('.chat-ai');
        await typeMessage(target, text);
    }
}

function showTyping(on) {
    $typing.classList.toggle('show', on);
    if (on) scrollChat();
}

function renderRecommendations(recs) {
    const rankCls = ['', 'rank-gold', 'rank-silver', 'rank-bronze'];
    $recList.innerHTML = recs.map(r => `
        <div class="card">
            <div class="rec-header">
                <div class="rec-title-row">
                    <span class="${rankCls[r.rank]}">#${r.rank}</span>
                    <strong style="font-size:1rem; color:#2A2A2A">${escHtml(r.title)}</strong>
                    <span class="badge badge-teal">${escHtml(r.category)}</span>
                </div>
                <span class="rec-rating">${r.rating}<small>/5</small></span>
            </div>
            <div class="rec-reason-row">
                <span class="check-mark">✓</span>
                <p class="rec-reason">${escHtml(r.reason)}</p>
            </div>
            <div class="evidence-box">${escHtml(r.evidence)}</div>
            ${r.risk ? `<div class="risk-box">⚠ ${escHtml(r.risk)}</div>` : ''}
        </div>
    `).join('');
    $recSection.style.display = 'block';
    scrollChat();
}

async function sendMessage(text) {
    if (!text.trim()) return;
    appendMsg('user', text);
    $input.value = '';
    $sendBtn.disabled = true;
    showTyping(true);

    try {
        const res = await fetch(CHAT_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ step: currentStep, message: text }),
        });
        const data = await res.json();
        currentStep = data.step;
        showTyping(false);
        await appendMsg('ai', data.reply);

        if (data.done && data.recommendations) {
            renderRecommendations(data.recommendations);
        }
    } catch (e) {
        showTyping(false);
        appendMsg('ai', '오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
        $sendBtn.disabled = false;
        $input.focus();
    }
}

$sendBtn.addEventListener('click', () => sendMessage($input.value));
$input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendMessage($input.value);
    }
});
document.querySelectorAll('.quick-chip').forEach(chip => {
    chip.addEventListener('click', () => sendMessage(chip.dataset.val));
});

// 스마트 챗봇 — 자유 문장 + 역질문 (SMART_CHAT_API_URL 사용)

// 슬롯별 역질문 버튼
const SLOT_BUTTONS = {
    person_count : ["2명", "3명", "4명", "5명 이상"],
    relationship : ["처음 만나는 사이", "친한 사이"],
    horror_tolerance : ["모두 괜찮음", "일부 민감함", "전체적으로 피하고 싶음"],
    budget : ["1인당 1만원", "1인당 2만원", "1인당 3만원 이상"],
    activity_level : ["조용한 활동 선호", "보통", "활발한 활동 선호"],
};

const $quickChips = document.querySelector('.quick-chips');

function updateQuickChips(buttons) {
    $quickChips.innerHTML = buttons
        .map(b => `<span class="quick-chip" data-val="${b}">${b}</span>`)
        .join('');

    // 새로 생긴 칩에도 클릭 이벤트 등록
    $quickChips.querySelectorAll('.quick-chip').forEach(chip => {
        chip.addEventListener('click', () => sendSmartMessage(chip.dataset.val));
    });
}

async function sendSmartMessage(text) {
    if (!text.trim()) return;
    appendMsg('user', text);
    $input.value = '';
    $sendBtn.disabled = true;
    showTyping(true);

    try {
        const res  = await fetch(SMART_CHAT_API_URL, {
            method : 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken' : getCookie('csrftoken'),
            },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();

        showTyping(false);
        await appendMsg('ai', data.reply);

        if (data.done) {
            // 모든 슬롯 완성 → 추천 결과 표시 + 초기 버튼으로 복원
            if (data.recommendations) renderRecommendations(data.recommendations);
            updateQuickChips(["4명", "처음 만나는 사이", "공포 싫어하는 사람 있어요", "1인당 2만원"]);
        } else {
            // 빠진 슬롯 → 역질문 버튼으로 교체
            const missing = (data.missing_slots || [])[0];
            const buttons = SLOT_BUTTONS[missing] || [];
            if (buttons.length) updateQuickChips(buttons);
        }

    } catch (e) {
        showTyping(false);
        appendMsg('ai', '오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
        $sendBtn.disabled = false;
        $input.focus();
    }
}

// 기존 단계별 칩 클릭을 스마트 챗봇으로 교체
document.querySelectorAll('.quick-chip').forEach(chip => {
    chip.addEventListener('click', () => sendSmartMessage(chip.dataset.val));
});

// 전송 버튼 / Enter를 스마트 챗봇으로 교체
$sendBtn.removeEventListener('click',  $sendBtn._handler);
$input.removeEventListener('keydown',  $input._handler);

$sendBtn.addEventListener('click', () => sendSmartMessage($input.value));
$input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendSmartMessage($input.value);
    }
});