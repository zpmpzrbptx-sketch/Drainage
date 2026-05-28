async function loadDecision(){
    const res = await fetch('/api/decision')
    const data = await res.json()

    document.getElementById("water").innerText = data.water
    document.getElementById("rain").innerText = data.rain
    document.getElementById("action").innerText = data.action
    document.getElementById("reason").innerText = data.reason
    document.getElementById("decisionTime").innerText = data.decision_time || "-"
    document.getElementById("strategyVersion").innerText = data.strategy_version || "-"
    document.getElementById("decisionId").innerText = data.decision_id || "-"
    document.getElementById("explainSignature").innerText = data.explain_signature || "-"
    const reviewWindow = typeof data.review_window_sec === "number" ? `${data.review_window_sec} 秒` : "-"
    document.getElementById("reviewWindow").innerText = reviewWindow

    const ruleHits = Array.isArray(data.rule_hits) ? data.rule_hits : []
    const ruleContainer = document.getElementById("ruleHits")
    ruleContainer.innerHTML = ""
    if (ruleHits.length === 0) {
        ruleContainer.innerHTML = '<p class="rule-empty">暂无规则命中数据</p>'
    } else {
        ruleHits.forEach((rule) => {
            const item = document.createElement("p")
            item.className = rule.hit ? "rule-hit" : "rule-miss"
            const mark = rule.hit ? "✓" : "✗"
            const weight = typeof rule.weight === "number" ? `（权重 ${rule.weight}）` : ""
            item.innerText = `${mark} ${rule.name}: ${rule.expr} ${weight}`
            ruleContainer.appendChild(item)
        })
    }

    const confidence = data.confidence || {}
    const confidenceScore =
        typeof confidence.score === "number" ? confidence.score.toFixed(2) : "-"
    document.getElementById("confidenceScore").innerText = confidenceScore

    const confidenceLevelElem = document.getElementById("confidenceLevel")
    const level = confidence.level || "-"
    confidenceLevelElem.innerText = level
    confidenceLevelElem.className = ""
    if (level === "高") confidenceLevelElem.classList.add("confidence-high")
    else if (level === "中") confidenceLevelElem.classList.add("confidence-medium")
    else if (level === "低") confidenceLevelElem.classList.add("confidence-low")

    const uncertaintyList = Array.isArray(confidence.uncertainty)
        ? confidence.uncertainty
        : []
    const uncertaintyContainer = document.getElementById("uncertaintyList")
    uncertaintyContainer.innerHTML = ""
    if (uncertaintyList.length === 0) {
        uncertaintyContainer.innerHTML = '<p class="rule-empty">暂无不确定性说明</p>'
    } else {
        uncertaintyList.forEach((text) => {
            const item = document.createElement("p")
            item.className = "uncertainty-item"
            item.innerText = `• ${text}`
            uncertaintyContainer.appendChild(item)
        })
    }

    const sensitivityList = Array.isArray(data.sensitivity) ? data.sensitivity : []
    const sensitivityContainer = document.getElementById("sensitivityList")
    sensitivityContainer.innerHTML = ""
    if (sensitivityList.length === 0) {
        sensitivityContainer.innerHTML = '<p class="rule-empty">暂无敏感性分析</p>'
    } else {
        sensitivityList.forEach((item) => {
            const row = document.createElement("div")
            row.className = "sensitivity-item"
            const statusClass = item.status === "已越界"
                ? "sensitivity-danger"
                : item.status === "临界区"
                    ? "sensitivity-warn"
                    : "sensitivity-safe"
            row.innerHTML = `
                <p><strong>${item.name}</strong> <span class="${statusClass}">${item.status}</span></p>
                <p>阈值距离: ${item.distance}</p>
                <p class="sensitivity-advice">${item.advice}</p>
            `
            sensitivityContainer.appendChild(row)
        })
    }

    const counterfactuals = Array.isArray(data.counterfactuals) ? data.counterfactuals : []
    const counterfactualContainer = document.getElementById("counterfactualList")
    counterfactualContainer.innerHTML = ""
    if (counterfactuals.length === 0) {
        counterfactualContainer.innerHTML = '<p class="rule-empty">暂无反事实推演</p>'
    } else {
        counterfactuals.forEach((item) => {
            const row = document.createElement("div")
            row.className = `counterfactual-item ${item.recommended ? "recommended" : ""}`
            const riskText = Number(item.risk_change) > 0
                ? `+${item.risk_change}`
                : `${item.risk_change}`
            row.innerHTML = `
                <p><strong>${item.name}</strong>${item.recommended ? "（推荐）" : ""}</p>
                <p>动作: [${(item.action || []).join(", ")}]</p>
                <p>风险变化: ${riskText}</p>
                <p>预估能耗: ${item.energy_change}</p>
                <p class="counterfactual-note">${item.note || ""}</p>
            `
            counterfactualContainer.appendChild(row)
        })
    }
}

loadDecision()
setInterval(loadDecision,3000)
