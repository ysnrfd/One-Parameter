[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Workflow,
    [string]$Plugin,
    [string]$BaseUrl = 'http://127.0.0.1:8765',
    [string]$OutputDirectory = '.\node-lab-results',
    [ValidateRange(1, 3600)]
    [int]$WaitSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$BaseUrl = $BaseUrl.TrimEnd('/')
$studioUri = [Uri]$BaseUrl
if ($studioUri.Scheme -ne 'http' -or $studioUri.Host -notin @('127.0.0.1', 'localhost', '[::1]', '::1')) {
    throw 'This example client requires the local loopback HTTP studio URL.'
}
$studioHeaders = @{}
$initialState = Invoke-RestMethod -Uri ($BaseUrl + '/api/graph/state')
if ($initialState.PSObject.Properties.Name -contains 'workspace_epoch') {
    $studioHeaders['X-Studio-Epoch'] = [string]$initialState.workspace_epoch
}

function Invoke-StudioAction {
    param([Parameter(Mandatory = $true)][hashtable]$Payload)
    $studioJson = ConvertTo-Json -InputObject $Payload -Depth 100 -Compress
    $studioBytes = [System.Text.Encoding]::UTF8.GetBytes($studioJson)
    Invoke-RestMethod -Uri ($BaseUrl + '/api/graph/action') -Method Post `
        -ContentType 'application/json; charset=utf-8' -Headers $studioHeaders -Body $studioBytes
}

if ($Plugin) {
    $pluginPath = (Resolve-Path -LiteralPath $Plugin).Path
    $pluginManifest = Get-Content -LiteralPath $pluginPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $registry = Invoke-RestMethod -Uri ($BaseUrl + '/api/graph/state') -Headers $studioHeaders
    $existing = @($registry.plugins | Where-Object { $_.id -eq $pluginManifest.id })
    if ($existing.Count -eq 0) {
        Invoke-StudioAction @{ action = 'plugin.install'; manifest = $pluginManifest } | Out-Null
    } else {
        Invoke-StudioAction @{ action = 'plugin.update'; plugin_id = $pluginManifest.id; manifest = $pluginManifest } | Out-Null
    }
    Invoke-StudioAction @{ action = 'plugin.enable'; plugin_id = $pluginManifest.id } | Out-Null
}

$workflowPath = (Resolve-Path -LiteralPath $Workflow).Path
$workflowGraph = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($workflowGraph.PSObject.Properties.Name -contains 'graph') {
    $workflowGraph = $workflowGraph.graph
}
$created = Invoke-StudioAction @{ action = 'graph.create'; graph = $workflowGraph }
$graphId = [string]$created.result.graph_id
$validated = Invoke-StudioAction @{ action = 'graph.validate'; graph_id = $graphId }
if (-not $validated.result.valid) {
    throw ('Workflow validation failed: ' + (ConvertTo-Json -InputObject $validated.result.errors -Depth 30))
}

$queued = Invoke-StudioAction @{
    action = 'graph.execute'
    graph_id = $graphId
    expected_revision = 1
    idempotency_key = [Guid]::NewGuid().ToString('N')
}
$jobId = [string]$queued.result.job_id
Write-Host ('Queued workflow ' + $graphId + ' as job ' + $jobId)
$workflowDeadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
$observedJob = $null
$rawState = $null
while ([DateTime]::UtcNow -lt $workflowDeadline) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri ($BaseUrl + '/api/graph/state') -Headers $studioHeaders
    $rawState = [string]$response.Content
    $observedState = $rawState | ConvertFrom-Json
    $matches = @($observedState.jobs | Where-Object { $_.id -eq $jobId })
    if ($matches.Count -eq 0) {
        throw 'The captured job is no longer retained in the server history.'
    }
    $observedJob = $matches[0]
    if ($observedJob.status -notin @('queued', 'running', 'cancelling')) {
        break
    }
    Start-Sleep -Milliseconds 250
}
if ($null -eq $observedJob -or $observedJob.status -in @('queued', 'running', 'cancelling')) {
    Invoke-StudioAction @{ action = 'job.cancel'; job_id = $jobId } | Out-Null
    throw ('Client wait expired; cancellation requested for ' + $jobId + '. Completed host effects remain recorded.')
}

$resultRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$jobDirectory = Join-Path -Path $resultRoot -ChildPath $jobId
New-Item -ItemType Directory -Path $jobDirectory -Force | Out-Null
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$stateFile = Join-Path -Path $jobDirectory -ChildPath 'graph-state.json'
[System.IO.File]::WriteAllText($stateFile, $rawState, $utf8NoBom)

foreach ($artifact in @($observedState.artifacts | Where-Object { $_.job_id -eq $jobId })) {
    if ($artifact.download_url -notmatch '^/api/graph/artifacts/[A-Za-z0-9_-]+$') {
        throw 'The server returned an unexpected artifact route.'
    }
    $filename = [System.IO.Path]::GetFileName([string]$artifact.filename)
    $artifactPath = Join-Path -Path $jobDirectory -ChildPath ($artifact.id + '-' + $filename)
    Invoke-WebRequest -UseBasicParsing -Uri ($BaseUrl + $artifact.download_url) -Headers $studioHeaders -OutFile $artifactPath
    $receivedHash = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($receivedHash -ne ([string]$artifact.sha256).ToLowerInvariant()) {
        throw ('Artifact SHA-256 mismatch: ' + $artifactPath)
    }
    if ((Get-Item -LiteralPath $artifactPath).Length -ne $artifact.byte_length) {
        throw ('Artifact length mismatch: ' + $artifactPath)
    }
    Write-Host ('Verified evidence ZIP: ' + $artifactPath)
}

Write-Host ('Job state: ' + $observedJob.status)
Write-Host ('Raw graph state: ' + $stateFile)
if ($observedJob.status -ne 'succeeded') {
    throw ('Execution ended as ' + $observedJob.status + ': ' + $observedJob.error)
}
